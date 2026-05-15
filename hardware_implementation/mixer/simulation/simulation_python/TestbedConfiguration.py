import logging
import random

import numpy as np

# logging format
fmt = "%(asctime)s %(filename)-15.15s:%(lineno)-5d %(levelname)-8s %(message)s"
dfmt = "%H:%M:%S"
logging.basicConfig(format=fmt, datefmt=dfmt)
logger = logging.getLogger("TestbedConfiguration")
logger.setLevel(logging.INFO)


class LinkModel:
    def __init__(self, config):
        self.Ptx_dBm = config["txpwr"]
        self.Pnoise_dBm = config["noise"]
        self.wavelength = config["wavelen"]  # default: c / f = 3e8 / 2.4e9
        self.fsplExponent = config["attenuation"]

    # Converts SINR to a receive probability. Arbitrary/custom function.
    def SINR2p(self, SINR_dB):
        return 1 / (1 + np.exp(-SINR_dB + 5))

    # Converts distance to antenna gain.
    # See free space path loss model: https://en.wikipedia.org/wiki/Free-space_path_loss
    def d2g(self, dist):
        return (self.wavelength / (4 * np.pi)) ** 2 / dist**self.fsplExponent

    # Converts antenna gain to distance.
    def g2d(self, gain):
        return ((self.wavelength / (4 * np.pi)) ** 2 / gain) ** (1 / self.fsplExponent)


class TestbedConfiguration:
    def __init__(self, configTestbed, configLinkmodel):
        # The linter does not like that approach.
        # for k, v in configTestbed.items():
        # 	setattr(self, k, v)

        self.areaW = configTestbed["areaW"]
        self.areaH = configTestbed["areaH"]
        self.numNodes = configTestbed["numNodes"]
        self.minDist = configTestbed["minDist"]
        self.minPRR = configTestbed["minPRR"]
        self.maxDist = configTestbed["maxDist"]
        self.linkmodel = LinkModel(configLinkmodel)

        if self.maxDist <= self.minDist:
            logger.error("dmin >= dmax, impossible parameter constellation")

        # generate nodes
        pos = np.zeros((self.numNodes, 2), dtype=int)
        d = np.zeros((self.numNodes, self.numNodes))

        # we need to initialize the first node outside the loop for the distance measure
        pos[0, :] = [random.randint(1, self.areaW), random.randint(1, self.areaH)]
        d[0, 0] = np.inf

        for k in range(1, self.numNodes):
            while True:
                # compute random position within area
                pos[k, :] = [
                    random.randint(1, self.areaW),
                    random.randint(1, self.areaH),
                ]
                # compute distance to all prior nodes
                d[k, :] = np.sqrt(
                    (pos[:, 0] - pos[k, 0]) ** 2 + (pos[:, 1] - pos[k, 1]) ** 2
                )
                # check whether the nearest node is out of range (> dmax)
                if min(d[k, 0:k]) > self.maxDist:
                    # distance to the nearest node
                    dd = np.amin(d[k, 0:k])
                    dd_idx = np.asarray(d[k, 0:k] == np.amin(d[k, 0:k])).nonzero()[0]
                    if len(dd_idx) > 1:
                        logger.info("Multiple nearest nodes. Take the first.")
                        dd_idx = dd_idx[0]
                    # calculate by how many percent dmax is exceeded
                    dd = 1 - self.maxDist / dd
                    # compute the position delta between k and the nearest node and scale down by dd
                    delta = dd * (pos[dd_idx, :] - pos[k, :])
                    # adjust the position accordingly
                    pos[k, :] = pos[k, :] + (np.sign(delta) * np.ceil(np.abs(delta)))
                    # recalculate distances
                    d[k, :] = np.sqrt(
                        (pos[:, 0] - pos[k, 0]) ** 2 + (pos[:, 1] - pos[k, 1]) ** 2
                    )
                    # the nearest node is now definately in range (< dmax)
                d[k, k] = np.inf

                # check whether the nearest node is too close (roll again the position)
                if min(d[k, 0:k]) >= self.minDist:
                    break
        self.nodes = pos

        # Generate link matrix (mirror lower triangle of d to the upper half).
        d = np.tril(d) + np.transpose(np.tril(d, -1))

        # Use inf in case of divide by zero.
        with np.errstate(divide="ignore", invalid="ignore"):
            self.linkMatrix = self.linkmodel.d2g(d)
            self.linkMatrix_dB = 10 * np.log10(self.linkMatrix)
