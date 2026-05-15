import argparse
import copy

import logging
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import PyPDF2
from fpdf import FPDF
from matplotlib import cm
from matplotlib.colors import ListedColormap
from MixerLogParser import MixerLogParser

# logging settings
logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter(
    fmt="%(filename)-15.15s:%(lineno)-5d %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)

# Log stats into file.
file_handler = None

# "Tableau 20" colors as RGB.
tableau20 = [
    (31, 119, 180),
    (174, 199, 232),
    (255, 127, 14),
    (255, 187, 120),
    (44, 160, 44),
    (152, 223, 138),
    (214, 39, 40),
    (255, 152, 150),
    (148, 103, 189),
    (197, 176, 213),
    (140, 86, 75),
    (196, 156, 148),
    (227, 119, 194),
    (247, 182, 210),
    (127, 127, 127),
    (199, 199, 199),
    (188, 189, 34),
    (219, 219, 141),
    (23, 190, 207),
    (158, 218, 229),
]

# Scale the RGB values to the [0, 1] range (matplotlib format).
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255.0, g / 255.0, b / 255.0)


# custom color map
def get_color_map(steps):
    old_cmap = cm.get_cmap("inferno", steps + 1)
    newcolors = old_cmap(np.linspace(0, 1, steps + 1))
    # old_cmap = cm.get_cmap('inferno', 25600)
    # newcolors = old_cmap(np.linspace(0, 1, 25600))
    white = np.array([0, 0, 0, 0])
    newcolors[-1] = white
    my_cmap = ListedColormap(newcolors)

    return my_cmap


# global variables for convenient access
class G:
    sync_stats = False


def extract_infos(file, cfg):
    results = {}

    with open(file, "r", encoding="utf-8") as log:
        lines = log.readlines()
        cur_rnd = None
        last_nodeid = None

        for linenumber, line in enumerate(lines):
            try:
                nodeid = int(line.split("|")[1].strip())
                if nodeid != last_nodeid:
                    last_nodeid = nodeid
                    cur_rnd = None

                if "starting round" in line:
                    cur_rnd = int(line.split("starting round")[1].split()[0])
                    # logger.debug(f'Found starting round {cur_rnd}')
                    continue

                if "preparing round" in line:
                    cur_rnd = int(line.split("preparing round")[1].split()[0])
                    # logger.debug(f'Found starting round {cur_rnd}')
                    continue

                if "sync statistics:" in line:
                    G.sync_stats = True
                    continue

                if cur_rnd is None:
                    # logger.debug(f'line without prior round information ... skipping line {repr(line)}')
                    continue

                nodeid_log = None
                for phy, log in cfg["NODE_MAPPING"]:
                    if phy == nodeid:
                        nodeid_log = log
                        break

                if nodeid_log is None:
                    logger.error(
                        f"NODE_MAPPING wrong or incomplete! No entry for node {nodeid}"
                    )
                    sys.exit(1)

                # At this point we have node "nodeid" which actually started a round.
                if nodeid not in results:
                    results[nodeid] = {}
                if cur_rnd not in results[nodeid]:
                    results[nodeid][cur_rnd] = {}

                if "rank_up_slot" in line:
                    # slots is a list of slots when the node's rank increased
                    slots = line.split("rank_up_slot=")[1].strip(" [];\n")
                    if len(slots) == 0:
                        continue
                    else:
                        slots = [int(s) for s in slots.split(";")]

                    # ext_slots is a list with MX_ROUND_LENGTH slots and those slots where the node's rank
                    # increased are filled with the actual rank of the node while the others are 0.
                    # ext_slots[0] is the 0th slot which is the round start before the first slot.
                    # ext_slots is larger than MX_ROUND_LENGTH to handle cases where the nodes reaches full
                    # rank at the end of a round.
                    rank = 0
                    ext_slots = [0] * (cfg["MX_ROUND_LENGTH"] + 5)
                    # ext_slots = [0] * (cfg['MX_ROUND_LENGTH'] + 1)
                    for slot in slots:
                        rank += 1
                        # try:
                        ext_slots[slot] = rank
                        # except IndexError:
                        # 	print(line)
                        # 	print(cfg['MX_ROUND_LENGTH'])

                    # fill slots between rank increases with the actual rank of the node
                    slot_full_rank = -1
                    last_rank = 0
                    for i, s in enumerate(ext_slots):
                        # determine in which slot full rank was reached
                        if s == cfg["MX_GENERATION_SIZE"]:
                            slot_full_rank = i
                        if s > last_rank:
                            last_rank = s
                        elif s < last_rank:
                            ext_slots[i] = last_rank

                    results[nodeid][cur_rnd]["rank_per_slot"] = ext_slots
                    results[nodeid][cur_rnd]["slot_full_rank"] = slot_full_rank
                    results[nodeid][cur_rnd]["final_rank"] = last_rank
                    results[nodeid][cur_rnd]["rank_up_slot"] = slots
                    continue

                if "| slot_full_rank:" in line:

                    data = line.split("slot_full_rank:", 1)[1]
                    value = int(data.strip())

                    if "slot_full_rank" in results[nodeid][cur_rnd]:
                        if value != results[nodeid][cur_rnd]["slot_full_rank"]:
                            logger.error(
                                f"inconsistent slot_full_rank value in line {linenumber+1}"
                            )
                    else:
                        results[nodeid][cur_rnd]["slot_full_rank"] = value

                if "| slot_off:" in line:

                    data = line.split("slot_off:", 1)[1]
                    value = int(data.strip())

                    results[nodeid][cur_rnd]["slot_off"] = value

                # packet trace
                if "# MXP" in line:

                    # line format: # MXP<fmt (%x)>: <slot (%04x)> <sender_id (%02x)> <flags (%02x)> [<info_vector>] <coding_vector> <payload>

                    data = line.split("# MXP", 1)[1]
                    fmt, data = data.split(":", 1)
                    data = data.split()

                    if int(fmt) != 1:
                        logger.warning(
                            "unknown MXP record format in line {} \"{} ...\"".format(
                            linenumber + 1, line.split(":", 1)[0])
                        )
                        continue

                    try:

                        slot   = int(data[0], base=16)
                        sender = int(data[1], base=16)
                        flags  = int(data[2], base=16)

                        if sender == nodeid_log:
                            if "info_type_per_slot" not in results[nodeid][cur_rnd]:
                                results[nodeid][cur_rnd]["info_type_per_slot"] = {}
                            results[nodeid][cur_rnd]["info_type_per_slot"][slot] = flags & 0x8f

                    except:
                        logger.warning(
                            f"invalid MXP record in line {linenumber+1} \"{line.strip()}\""
                        )

                    continue

                # aggregate trace
                if "# MXA" in line:

                    # line format:
                    # # MXA<fmt (%x)>: <slot (%04x)> <phase+agg (%03x)> <progress (%02x)>:<progress flags (%x)>
                    # / <tx_phase_data> <local_phase_data>
                    #
            		# <phase+agg>:
            		# 0x000...0x3FF: phase = 0 (Collect), 0b0000_00xx_xxxx = list length
            		# 0x400...0x7FF: phase = 1 (Prepare), 0b00xx_xxxx_xxxx = proposal
            		# 0x800...0xBFF: phase = 2 (Accept),  0b00xx_xxxx_xxxx = proposal
            		#
            		# <progress> = progress of current phase (number of set progress flags)
            		#
            		# <tx_phase_data> in phase 0:
            		# <node (%02x)>:<prio (%02x)> for each list entry
            		#
            		# <tx_phase_data> in phase 1:
            		# <max_accepted_proposal (%03x)> <value (%x)>
            		#
            		# <tx_phase_data> in phase 2:
            		# <max_observed_proposal (%03x)> <value (%x)>
            		#
            		# <local_phase_data> in phase 1+2:
            		# < min_proposal (%03x)> <accepted_proposal (%03x)> <max_observed_proposal> <max_accepted_proposal>
            		#
                    #
                    # examples:
                    # MXA1: 0000 001 01:01 / 00:10
                    # MXA1: 0015 404 03:07 / fff 03 004 fff fff 000
                    # MXA1: 0021 804 03:07 / 004 03 004 004 004 004

                    data = line.split("# MXA", 1)[1]
                    fmt, data = data.split(":", 1)
                    data = data.split()

                    if int(fmt) != 1:
                        logger.warning(
                            "unknown MXA record format in line {} \"{} ...\"".format(
                            linenumber + 1, line.split(":", 1)[0])
                        )
                        continue

                    try:

                        slot        = int(data[0], base=16)
                        phase       = int(data[1], base=16)
                        proposal    = phase & 0x3ff
                        phase       = phase >> 10
                        progress    = int(data[2].split(":")[0], base=16)

                        if 0 == phase:
                            proposal          = nodeid_log
                            min_proposal      = None
                            accepted_proposal = None
                            max_op = max_ap   = None
                        else:
                            min_proposal      = int(data[6], base=16)
                            x                 = int(data[7], base=16)
                            accepted_proposal = None if 0xfff == x else x
                            max_op            = int(data[8], base=16)
                            x                 = int(data[9], base=16)
                            max_ap            = None if 0xfff == x else x

                        if "agg_per_slot" not in results[nodeid][cur_rnd]:
                            results[nodeid][cur_rnd]["agg_per_slot"] = {}

                        results[nodeid][cur_rnd]["agg_per_slot"][slot] = {
                            "phase": phase, "progress": progress, "proposal": proposal,
                            "min_proposal": min_proposal, "accepted_proposal": accepted_proposal,
                            "max_observed_proposal": max_op, "max_accepted_proposal": max_ap
                        }

                    except:
                        logger.warning(
                            f"invalid MXA record in line {linenumber+1} \"{line.strip()}\""
                        )

                    continue

                # aggregate statistics
                if "aggregate.slot_" in line:

                    if "agg" not in results[nodeid][cur_rnd]:
                        results[nodeid][cur_rnd]["agg"] = {}

                    data = line.split("aggregate.slot_", 1)[1]
                    data = data.split(":", 1)
                    field = "slot_" + data[0].strip()
                    value = int(data[1].strip())

                    results[nodeid][cur_rnd]["agg"][field] = value
                    continue

                if G.sync_stats:
                    # if "resync_group" in line:
                    if "resync_id" in line:
                        # NOTE: IDs in resync_group are logical IDs.
                        # groups = line.split('resync_group=')[1].strip(' [];\n')
                        groups = line.split("resync_id=")[1].strip(" [];\n")
                        if len(groups) == 0:
                            groups = [nodeid_log]
                        else:
                            groups = [int(s) for s in groups.split(";")]
                            groups.insert(0, nodeid_log)

                        # results[nodeid][cur_rnd]['resync_group'] = groups
                        results[nodeid][cur_rnd]["resync_id"] = groups
                        continue

                    if "resync_countdown" in line:
                        countdowns = line.split("resync_countdown=")[1].strip(
                            " [];\n"
                        )
                        if len(countdowns) == 0:
                            countdowns = [cfg["SYNC_COUNTDOWN"]]
                        else:
                            countdowns = [
                                int(s) for s in countdowns.split(";")
                            ]
                            countdowns.insert(0, cfg["SYNC_COUNTDOWN"])

                        results[nodeid][cur_rnd]["resync_countdown"] = (
                            countdowns
                        )
                        continue

                # discovery information when tracing is activated
                # NOTE: compatibility for older logs
                res = re.search(
                    r"discovery exit slot: (?P<discoveryExit>\d+) \(density: (?P<discoveryDensity>\d+), wake up: (?P<wakeUp>\d+)\)",
                    line,
                )
                if res:
                    discoveryExit = int(res.group("discoveryExit"))
                    discoveryDensity = int(res.group("discoveryDensity"))
                    wakeUp = int(res.group("wakeUp"))
                    results[nodeid][cur_rnd]["discoveryExit"] = discoveryExit
                    results[nodeid][cur_rnd]["discoveryDensity"] = (
                        discoveryDensity
                    )
                    results[nodeid][cur_rnd]["wakeUp"] = wakeUp
                    continue

                # round summary
                # NOTE: compatibility for older logs
                res = re.search(
                    r"rank=(?P<rank>\d+) dec=(?P<decoded>\d+) !dec=(?P<notDecoded>\d+) weak=(?P<weak>\d+) wrong=(?P<wrong>\d+)",
                    line,
                )
                if not res:
                  res = re.search(
                      r"rank=(?P<rank>\d+) dec=(?P<decoded>\d+) notDec=(?P<notDecoded>\d+) weak=(?P<weak>\d+) wrong=(?P<wrong>\d+)",
                      line,
                  )
                if res:
                    # we already retrieve the rank from rank_up_slot list
                    # rank				= int(res.group('rank'))
                    decoded = int(res.group("decoded"))
                    # notDecoded = int(res.group("notDecoded"))
                    weak = int(res.group("weak"))
                    # wrong = int(res.group("wrong"))

                    results[nodeid][cur_rnd]["decoded"] = decoded + weak
                    continue

                # NOTE: compatibility for older logs
                res = re.search(
                    r"rank=(?P<rank>\d+) dec=(?P<decoded>\d+) notDec=(?P<notDecoded>\d+) wrong=(?P<wrong>\d+)",
                    line,
                )
                if res:
                    # we already retrieve the rank from rank_up_slot list
                    # rank				= int(res.group('rank'))
                    decoded = int(res.group("decoded"))
                    # notDecoded = int(res.group("notDecoded"))
                    # wrong = int(res.group("wrong"))

                    results[nodeid][cur_rnd]["decoded"] = decoded
                    continue

                # NOTE: compatibility for older logs
                res = re.search(
                    r"rank=(?P<rank>\d+) decoded=(?P<decoded>\d+) discovery_exit=(?P<discoveryExit>\d+) discovery_density=(?P<discoveryDensity>\d+)",
                    line,
                )
                if res:
                    # we already retrieve the rank from rank_up_slot list
                    # rank				= int(res.group('rank'))
                    decoded = int(res.group("decoded"))
                    discoveryExit = int(res.group("discoveryExit"))
                    discoveryDensity = int(res.group("discoveryDensity"))

                    results[nodeid][cur_rnd]["decoded"] = decoded
                    results[nodeid][cur_rnd]["discoveryExit"] = discoveryExit
                    results[nodeid][cur_rnd]["discoveryDensity"] = (
                        discoveryDensity
                    )
                    continue

                # NOTE: compatibility for older logs
                res = re.search(
                    r"rank=(?P<rank>\d+) decoded=(?P<decoded>\d+)", line
                )
                if res:
                    # we already retrieve the rank from rank_up_slot list
                    # rank	= int(res.group('rank'))
                    decoded = int(res.group("decoded"))

                    results[nodeid][cur_rnd]["decoded"] = decoded
                    continue

                res = re.search(r"decoded=(?P<decoded>\d+)\n", line)
                if res:
                    decoded = int(res.group("decoded"))

                    results[nodeid][cur_rnd]["decoded"] = decoded
                    continue

                res = re.search(
                    r"discovery_density: (?P<discoveryDensity>\d+)\n", line
                )
                if res:
                    discoveryDensity = int(res.group("discoveryDensity"))

                    results[nodeid][cur_rnd]["discoveryDensity"] = (
                        discoveryDensity
                    )
                    continue

                res = re.search(
                    r"discovery_exit_slot: (?P<discoveryExit>\d+)\n", line
                )
                if res:
                    discoveryExit = int(res.group("discoveryExit"))

                    results[nodeid][cur_rnd]["discoveryExit"] = discoveryExit
                    continue

            except ValueError as ve:
                # logger.info(f'ValueError {ve}. Skipping line {repr(line)}')
                logger.info(f"ValueError {ve}. Skipping line {repr(line)}")
                continue

    return results


def node_discoveryNeighbors_violin(mlp, results, force=False):

    outputfile = mlp.plotPath / "node_discoveryNeighbors_violin.pdf"
    if outputfile.exists() and not force:
        logger.info(f"{outputfile} already exists. Skipping plot...")
        return outputfile

    try:

        num_rounds = min([len(rnds) for rnds in results.values()])
        neighbors_per_node_all_rounds = []

        for node in sorted(results.keys()):
            rnds = results[node]
            neighbors_one_node_all_rounds = []
            for rnd, metrics in rnds.items():
                try:
                    neighbors_one_node_all_rounds.append(
                        metrics["discoveryDensity"]
                    )
                except KeyError as ke:
                    logger.debug(
                        f"Missing information {ke} for node {node} in round {rnd}"
                    )

            if len(neighbors_one_node_all_rounds) == 0:
                logger.info(
                    f'Couldn\'t find information about "discoveryDensity" for node {node}.'
                )
                return None

            neighbors_per_node_all_rounds.append(neighbors_one_node_all_rounds)

        # plt.subplots(1, 1, figsize=(0.8 * mlp.exp_config['MX_NUM_NODES'], 5))
        plt.figure(figsize=(0.8 * mlp.exp_config["MX_NUM_NODES"], 5))
        plt.violinplot(
            neighbors_per_node_all_rounds,
            showmeans=True,
            showmedians=False,
            showextrema=False,
        )

        percs = []
        for n in neighbors_per_node_all_rounds:
            p = np.percentile(n, [5, 95])
            percs.append(p)

        for i, p in enumerate(percs, 1):
            plt.hlines(p[0], xmin=i - 0.1, xmax=i + 0.1)
            plt.hlines(p[1], xmin=i - 0.1, xmax=i + 0.1)

        plt.xticks(
            ticks=range(1, mlp.exp_config["MX_NUM_NODES"] + 1),
            labels=sorted(results.keys()),
        )
        plt.xlabel("Logical Node ID")
        plt.ylabel("Neighbors After Discovery")
        plt.title(
            f'{mlp.exp_config["MX_PHY_NAME"]}, number of neighbors after discovery phase over {num_rounds} rounds\n({mlp.basepath.name})'
        )

        plt.gcf().savefig(outputfile, bbox_inches="tight")
        plt.close()

    except:
        logger.error("node_discoveryNeighbors_violin() failed")
        outputfile.unlink(missing_ok=True)
        outputfile = None

    return outputfile


def sync_groups_all_violin(mlp, results, force=False):
    outputfile = mlp.plotPath / "sync_groups_all_violin.pdf"
    if outputfile.exists() and not force:
        logger.info(f"{outputfile} already exists. Skipping plot...")
        return outputfile

    # sync_per_round = {}

    # for node, rnds in results.items():
    # 	for rnd_id, rnd in rnds.items():
    # 		if not rnd_id in sync_per_round:
    # 			sync_per_round[rnd_id] = {"groups": [], "countdown": []}
    # 		sync_per_round[rnd_id]["groups"].append(rnd["resync_group"][-1])
    # 		sync_per_round[rnd_id]["countdown"].append(rnd["resync_countdown"][-1])

    # sync_rounds_total = 0
    # successful_sync_rounds = 0
    # unsuccessful_sync_rounds = []
    # countdowns = []
    # all_sync_countdowns = []
    # groups = []

    # for rnd, data in sync_per_round.items():
    # 	sync_rounds_total += 1
    # 	if len(set(data["groups"])) == 1:
    # 		successful_sync_rounds += 1
    # 		countdowns.extend(data["countdown"])
    # 		all_sync_countdowns.append(np.min(data["countdown"]))
    # 		groups.append(data["groups"][0])
    # 	else:
    # 		unsuccessful_sync_rounds.append(rnd)

    # plt.figure(figsize=(15, 0.3 * mlp.exp_config['MX_NUM_NODES']))
    # plt.violinplot(all_sync_countdowns, showmeans=True, showmedians=False, showextrema=False)

    sync_all_nodes_all_rounds = []

    for node in sorted(results.keys()):
        rnds = results[node]
        sync_one_node_all_rounds = []
        for rnd, metrics in rnds.items():
            try:
                sync_one_node_all_rounds.append(
                    (
                        metrics["resync_group"],
                        metrics["resync_countdown"],
                        node,
                        rnd,
                    )
                )
            except KeyError as ke:
                logger.debug(
                    f"Missing information {ke} for node {node} in round {rnd}"
                )

        if len(sync_one_node_all_rounds) == 0:
            logger.info(
                f'Couldn\'t find information about "resync_group" or "resync_countdown" for node {node}.'
            )
            return None

        sync_all_nodes_all_rounds.append(sync_one_node_all_rounds)

    violindata = []

    for node_rnds in sync_all_nodes_all_rounds:
        violindata.append(
            [
                mlp.exp_config["SYNC_COUNTDOWN"] - np.min(rnd[1])
                for rnd in node_rnds
            ]
        )

    plt.figure(figsize=(15, 0.3 * mlp.exp_config["MX_NUM_NODES"]))
    plt.violinplot(
        violindata, showmeans=True, showmedians=False, showextrema=False
    )

    percs = []
    for n in violindata:
        p = np.percentile(n, [99.0, 99.9, 100.0])  # , axis=1)
        # p = np.append(p, np.max(n))
        # p.append(np.max(n))
        percs.append(p)

    for i, p in enumerate(percs, 1):
        # plt.hlines(p[0], xmin=i-0.1, xmax=i+0.1)
        # plt.hlines(p[1], xmin=i-0.1, xmax=i+0.1)
        for pe in p:
            plt.hlines(pe, xmin=i - 0.1, xmax=i + 0.1)

    # bot,top = plt.ylim()
    # plt.ylim([bot - 24, top])

    plt.grid(axis="y", alpha=0.5, lw=0.5)

    plt.xlabel("Logical Node ID")
    plt.ylabel("Time [sync slots]")
    plt.xticks(
        ticks=np.arange(1, mlp.exp_config["MX_NUM_NODES"] + 1),
        labels=np.arange(mlp.exp_config["MX_NUM_NODES"]),
    )
    plt.title("Per node time to sync distribution")

    plt.tight_layout()
    plt.gcf().savefig(outputfile, bbox_inches="tight")
    plt.close()

    return outputfile


def sync_groups_time_line_all(mlp, results, force=False):
    outputfile = mlp.plotPath / "sync_groups_time_line_all.pdf"
    if outputfile.exists() and not force:
        logger.info(f"{outputfile} already exists. Skipping plot...")
        return outputfile

    num_rounds = min([len(rnds) for rnds in results.values()])

    sync_all_nodes_all_rounds = []

    for node in sorted(results.keys()):
        rnds = results[node]
        sync_one_node_all_rounds = []
        for rnd, metrics in rnds.items():
            try:
                sync_one_node_all_rounds.append(
                    (
                        metrics["resync_group"],
                        metrics["resync_countdown"],
                        node,
                        rnd,
                    )
                )
            except KeyError as ke:
                logger.debug(
                    f"Missing information {ke} for node {node} in round {rnd}"
                )

        if len(sync_one_node_all_rounds) == 0:
            logger.info(
                f'Couldn\'t find information about "resync_group" or "resync_countdown" for node {node}.'
            )
            return None

        sync_all_nodes_all_rounds.append(sync_one_node_all_rounds)

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    for rnd in range(num_rounds - 1):
        for node in sync_all_nodes_all_rounds:
            x = node[rnd][1]
            y = node[rnd][0]
            # n = node[rnd][2]
            # r = node[rnd][3]
            # print(f'node={n} round={r} x={x} y={y}')

            y_tmp = [y[0]] * mlp.exp_config["SYNC_COUNTDOWN"]
            idx = 1
            for i in range(len(x) - 1):
                t = x[idx]
                g = y[idx]
                y_tmp[mlp.exp_config["SYNC_COUNTDOWN"] - t :] = [g] * t
                idx += 1
            # print(y_tmp)

            ax.plot(y_tmp)
            ax.set_xlabel("Time [sync slots]")
            ax.set_ylabel("Physical Node ID")

            ax.set_xlim(0, mlp.exp_config["SYNC_COUNTDOWN"])
            # ax.set_xticks(range(200))
            # ax.set_xticklabels(range(200,0))
            ax.set_yticks(range(0, mlp.exp_config["MX_NUM_NODES"]))
            ax.set_yticklabels(sorted(results.keys()))

    plt.gcf().savefig(outputfile, bbox_inches="tight")
    plt.close()
    return outputfile


def sync_stats(mlp, results, force=False):
    sync_per_round = {}

    for node, rnds in results.items():
        for rnd_id, rnd in rnds.items():
            if rnd_id not in sync_per_round:
                sync_per_round[rnd_id] = {"groups": [], "countdown": []}
            sync_per_round[rnd_id]["groups"].append(rnd["resync_group"][-1])
            sync_per_round[rnd_id]["countdown"].append(
                rnd["resync_countdown"][-1]
            )

    sync_rounds_total = 0
    successful_sync_rounds = 0
    unsuccessful_sync_rounds = []
    countdowns = []
    all_sync_countdowns = []
    groups = []

    # group leader distr.
    # sync reliability
    # mean (max/min) time to sync
    for rnd, data in sync_per_round.items():
        sync_rounds_total += 1
        if len(set(data["groups"])) == 1:
            successful_sync_rounds += 1
            countdowns.extend(data["countdown"])
            all_sync_countdowns.append(np.min(data["countdown"]))
            groups.append(data["groups"][0])
        else:
            unsuccessful_sync_rounds.append(rnd)
    logger.info(f"Unsuccessful sync rounds: {unsuccessful_sync_rounds}")

    logger.addHandler(file_handler)
    logger.info(
        f'Mean time all nodes sync: {mlp.exp_config["SYNC_COUNTDOWN"] - np.mean(all_sync_countdowns)}'
    )
    logger.info(
        f'Fastest time all nodes sync: {mlp.exp_config["SYNC_COUNTDOWN"] - np.max(all_sync_countdowns)}'
    )
    logger.info(
        f'Slowest time all nodes sync: {mlp.exp_config["SYNC_COUNTDOWN"] - np.min(all_sync_countdowns)}'
    )
    logger.info(
        f'Mean time per node sync: {mlp.exp_config["SYNC_COUNTDOWN"] - np.mean(countdowns)}'
    )
    logger.info(
        f"Sync reliabilty (successful sync rounds): {successful_sync_rounds / sync_rounds_total * 100}%"
    )
    logger.info(
        f'Time all nodes sync at 99% / 99.9% reliability: {mlp.exp_config["SYNC_COUNTDOWN"] - np.percentile(all_sync_countdowns, 1.0)} / {mlp.exp_config["SYNC_COUNTDOWN"] - np.percentile(all_sync_countdowns, 0.1)}'
    )
    logger.removeHandler(file_handler)

    # plt.hist(groups)
    # plt.savefig(mlp.plotPath / "sync_leader_histogram.pdf", bbox_inches='tight')
    # plt.close()

    # 1639152312.005177 |  1 | sync statistics:
    # 1639152312.005714 |  1 | resyncs: 1
    # 1639152312.005726 |  1 | tx_cnt: 10
    # 1639152312.009473 |  1 | rx_timeout: 10
    # 1639152312.009229 |  1 | rx_invalid_len: 0
    # 1639152312.009415 |  1 | rx_crc_error: 3
    # 1639152312.013611 |  1 | rx_success: 28
    # 1639152312.012846 |  1 | countdown_first_tx: 112
    # 1639152312.017490 |  1 | is_sync_leader: 0
    # 1639152312.017245 |  1 | rx_duration: 39059us
    # 1639152312.017865 |  1 | sync_group: 7
    # 1639152312.021464 |  1 | sync_countdown: 114


def sync_groups_time_line_single(mlp, results, force=False):
    outputfile = mlp.plotPath / "sync_groups_time_line_single.pdf"
    if outputfile.exists() and not force:
        logger.info(f"{outputfile} already exists. Skipping plot...")
        return outputfile

    # num_rounds = min([len(rnds) for rnds in results.values()])

    sync_all_nodes_all_rounds = []

    for node in sorted(results.keys()):
        rnds = results[node]
        sync_one_node_all_rounds = []
        for rnd, metrics in rnds.items():
            try:
                sync_one_node_all_rounds.append(
                    (
                        metrics["resync_group"],
                        metrics["resync_countdown"],
                        node,
                        rnd,
                    )
                )
            except KeyError as ke:
                logger.debug(
                    f"Missing information {ke} for node {node} in round {rnd}"
                )

        if len(sync_one_node_all_rounds) == 0:
            logger.info(
                f'Couldn\'t find information about "resync_group" or "resync_countdown" for node {node}.'
            )
            return None

        sync_all_nodes_all_rounds.append(sync_one_node_all_rounds)

    rounds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # -2 to adjust for stripping the first round and 0 indexing
    # rounds = [117-2, 132-2, 186-2, 228-2, 243-2, 374-2, 384-2, 444-2]

    fig, axarr = plt.subplots(len(rounds), 1, figsize=(20, len(rounds) * 4))

    for enu, rnd in enumerate(rounds):
        ax = axarr[enu]

        for node in sync_all_nodes_all_rounds:
            x = node[rnd][1]
            y = node[rnd][0]
            # n = node[rnd][2]
            r = node[rnd][3]
            # print(f'node={n} round={r} x={x} y={y}')

            y_tmp = [y[0]] * mlp.exp_config["SYNC_COUNTDOWN"]
            idx = 1
            for i in range(len(x) - 1):
                t = x[idx]
                g = y[idx]
                y_tmp[mlp.exp_config["SYNC_COUNTDOWN"] - t :] = [g] * t
                idx += 1
            # print(y_tmp)

            ax.plot(y_tmp)
            ax.set_xlabel(f"Time [sync slots] - round {r}")
            ax.set_ylabel("Physical Node ID")

            ax.set_xlim(0, mlp.exp_config["SYNC_COUNTDOWN"])
            # ax.set_xticks(range(200))
            # ax.set_xticklabels(range(200,0))
            ax.set_yticks(range(0, mlp.exp_config["MX_NUM_NODES"]))
            ax.set_yticklabels(sorted(results.keys()))

    plt.gcf().savefig(outputfile, bbox_inches="tight")
    plt.close()
    return outputfile


def sync_leader_distribution(mlp, results, force=False):
    outputfile = mlp.plotPath / "sync_leader_distribution.pdf"
    if outputfile.exists() and not force:
        logger.info(f"{outputfile} already exists. Skipping plot...")
        return outputfile

    sync_per_round = {}

    for node, rnds in results.items():
        for rnd_id, rnd in rnds.items():
            if rnd_id not in sync_per_round:
                sync_per_round[rnd_id] = {"groups": [], "countdown": []}
            sync_per_round[rnd_id]["groups"].append(rnd["resync_group"][-1])
            sync_per_round[rnd_id]["countdown"].append(
                rnd["resync_countdown"][-1]
            )

    sync_rounds_total = 0
    successful_sync_rounds = 0
    unsuccessful_sync_rounds = []
    countdowns = []
    all_sync_countdowns = []
    groups = []

    for rnd, data in sync_per_round.items():
        sync_rounds_total += 1
        if len(set(data["groups"])) == 1:
            successful_sync_rounds += 1
            countdowns.extend(data["countdown"])
            all_sync_countdowns.append(np.min(data["countdown"]))
            groups.append(data["groups"][0])
        else:
            unsuccessful_sync_rounds.append(rnd)

    plt.figure(figsize=(15, 0.3 * mlp.exp_config["MX_NUM_NODES"]))

    # https://stackoverflow.com/a/27084005
    bins = np.arange(mlp.exp_config["MX_NUM_NODES"] + 1) - 0.5
    plt.hist(groups, bins=bins, density=True)
    # alternative
    # bins = range(mlp.exp_config['MX_NUM_NODES'] + 1)
    # h,e = np.histogram(groups, bins=bins)
    # plt.bar(range(len(bins)-1), h, width=1, edgecolor='k')

    plt.xlabel("Logical Node ID")
    plt.ylabel("Probability")
    plt.xticks(ticks=np.arange(mlp.exp_config["MX_NUM_NODES"]))
    plt.title("Sync leader histogram")

    plt.tight_layout()
    plt.gcf().savefig(outputfile, bbox_inches="tight")
    plt.close()

    return outputfile


def slots_node_rank_heatmap_func(mlp, results, func, funcName, force=False):
    outputfile = mlp.plotPath / f"slots_node_rank_heatmap_{funcName}.pdf"
    if outputfile.exists() and not force:
        logger.info(
            f"{outputfile} already exists. Delete the file to redo the plot."
        )
        return outputfile

    # rounds completed by all nodes
    num_rounds = min([len(rnds) for rnds in results.values()])

    # applies func to the node's ranks per slot
    data = []
    for nodeid in sorted(results.keys()):
        rank_per_slot = []
        # Groups the same slots across all rounds for every node.
        for slot in zip(
            *[
                rnd["rank_per_slot"]
                for rnd in results[nodeid].values()
                if "rank_per_slot" in rnd
            ]
        ):
            rank_per_slot.append(func(slot))
        data.append(rank_per_slot)

    plt.figure(figsize=(15, 0.3 * mlp.exp_config["MX_NUM_NODES"]))
    pos = plt.imshow(
        data,
        interpolation="none",
        aspect="auto",
        cmap="inferno",
        vmin=0,
        vmax=mlp.exp_config["MX_GENERATION_SIZE"],
    )
    # pos = plt.imshow(data, interpolation='none', aspect='auto',
    # 				 cmap=get_color_map(mlp.exp_config['MX_GENERATION_SIZE']),
    # 				 vmin=0, vmax=mlp.exp_config['MX_GENERATION_SIZE'])
    plt.colorbar(pos, pad=0.01)

    plt.yticks(
        ticks=np.arange(len(results.keys())), labels=sorted(results.keys())
    )
    plt.ylim(len(results.keys()) - 0.5, -0.5)
    plt.xlabel("slots")
    plt.ylabel("nodes")
    plt.title(
        f'{mlp.exp_config["MX_PHY_NAME"]}, {funcName} ranks per slot over {num_rounds} rounds\n({mlp.basepath.name})'
    )

    plt.tight_layout()
    plt.gcf().savefig(outputfile, bbox_inches="tight")
    plt.close()
    return outputfile


def slots_node_rank_heatmap_rounds(mlp, results, rounds, force=False):
    outputfile = (
        mlp.plotPath
        / f"slots_node_rank_heatmap_rounds_{rounds[0]}_{rounds[1]}.pdf"
    )
    if outputfile.exists() and not force:
        logger.info(
            f"{outputfile} already exists. Delete the file to redo the plot."
        )
        return outputfile

    rnd_intersection = []
    for rnds in results.values():
        rnd_intersection = list(rnds)
        break

    num_rounds = len(rnd_intersection)
    # round_selection = random.sample(rnd_intersection, k=min(num_rounds, rounds))
    round_selection = rnd_intersection[rounds[0] : rounds[1]]
    logger.info(f"Plotting {len(round_selection)} rounds: {round_selection}")

    # rounds_to_plot = min(num_rounds, rounds)
    rounds_to_plot = len(round_selection)
    fig, axarr = plt.subplots(
        rounds_to_plot,
        1,
        figsize=(10, rounds_to_plot * mlp.exp_config["MX_NUM_NODES"] * 0.15),
    )

    if num_rounds == 1:
        rnd_axs = [(round_selection[0], axarr)]
    else:
        rnd_axs = zip(round_selection, axarr)

    for rnd, ax in rnd_axs:
        data = []
        for nodeid in sorted(results.keys()):
            data.append(results[nodeid][rnd]["rank_per_slot"])

        pos = ax.imshow(
            data,
            interpolation="none",
            aspect="auto",
            cmap="inferno",
            vmin=0,
            vmax=mlp.exp_config["MX_GENERATION_SIZE"],
        )
        fig.colorbar(pos, ax=ax, pad=0.01)

        ax.set_yticks(np.arange(len(results.keys())))
        ax.set_yticklabels(sorted(results.keys()))
        ax.set_ylim(len(results.keys()) - 0.5, -0.5)

        ax.set_title(
            f'{mlp.exp_config["MX_PHY_NAME"]}, Round {rnd}\n({mlp.basepath.name})'
        )

    plt.tight_layout()
    plt.gcf().savefig(outputfile, bbox_inches="tight")
    plt.close()
    return outputfile


def slots_rank_linePerNode_func(mlp, results, func, funcName, force=False):
    outputfile = mlp.plotPath / f"slots_rank_linePerNode_{funcName}.pdf"
    if outputfile.exists() and not force:
        logger.info(
            f"{outputfile} already exists. Delete the file to redo the plot."
        )
        return outputfile

    num_rounds = min([len(rnds) for rnds in results.values()])

    nodes_per_row = 7
    cols = min(mlp.exp_config["MX_NUM_NODES"], nodes_per_row)
    rows = math.ceil(mlp.exp_config["MX_NUM_NODES"] / nodes_per_row)
    fig, axarr = plt.subplots(
        rows, cols, sharex=True, sharey=True, figsize=(cols * 2, rows * 2)
    )

    # flatten axarr
    if isinstance(axarr[0], np.ndarray):
        axarr = [item for sublist in axarr for item in sublist]

    data = {}
    for nodeid in sorted(results.keys()):
        rank_per_slot = []
        # Groups the same slots across all rounds for every node.
        for slot in zip(
            *[
                rnd["rank_per_slot"]
                for rnd in results[nodeid].values()
                if "rank_per_slot" in rnd
            ]
        ):
            rank_per_slot.append(func(slot))
        data[nodeid] = rank_per_slot

    # This loop iterates only len(results.keys()) often even if axarr is bigger.
    for nodeid, ax in zip(sorted(results.keys()), axarr):
        ax.set_xlabel("slots")
        ax.set_ylabel("rank")
        ax.set_title(f"node {nodeid}")
        # ax.text(0.05, 0.9, f'node {nodeid}', transform=ax.transAxes)
        ax.set_ylim(0, mlp.exp_config["MX_GENERATION_SIZE"])
        y = data[nodeid]
        x = range(len(y))
        ax.plot(x, y, "-k")

    # common title for all subplots
    plt.suptitle(
        f'{mlp.exp_config["MX_PHY_NAME"]}, {funcName} rank per slot over {num_rounds} rounds\n({mlp.basepath.name})',
        x=0.5,
        y=1.03,
    )
    plt.tight_layout()
    plt.gcf().savefig(outputfile, bbox_inches="tight")
    plt.close()
    return outputfile


def node_fullRankSlot_reliability_violin(mlp, results, force=False):

    outputfile = mlp.plotPath / "node_fullRankSlot_reliability_violin.pdf"
    if outputfile.exists() and not force:
        logger.info(f"{outputfile} already exists. Skipping plot...")
        return outputfile

    num_rounds = min([len(rnds) for rnds in results.values()])
    full_rank_per_node_all_rounds = []
    data_decoded_per_node_all_rounds = []
    slot_off_per_node_all_rounds = []

    # {nodeid: {round: {slots, ...}}}
    last_num_rounds = None
    for node in sorted(results.keys()):
        rnds = results[node]

        if last_num_rounds is None:
            last_num_rounds = len(rnds.keys())
        elif last_num_rounds != len(rnds.keys()):
            logger.debug(
                f"Inconsistency in number of rounds. Node {node} has {len(rnds.keys())} rounds compared to {last_num_rounds} rounds of previous nodes."
            )

        full_rank_one_node_all_rounds = []
        data_decoded_one_node_all_rounds = []
        slot_off_one_node_all_rounds = []
        # for metrics in rnds.values():
        for rnd, metrics in rnds.items():
            try:
                # Not reaching full rank will not affect the full rank distribution plot but reliability.
                if metrics["slot_full_rank"] >= 0:
                    full_rank_one_node_all_rounds.append(
                        metrics["slot_full_rank"]
                    )
                if "decoded" not in metrics:
                    data_decoded_one_node_all_rounds.append(-1)
                else:
                    data_decoded_one_node_all_rounds.append(metrics["decoded"])
                slot_off_one_node_all_rounds.append(metrics["slot_off"])
            except KeyError as ke:
                logger.debug(
                    f"Missing information for node {node} in round {rnd}: {ke}"
                )

        if len(full_rank_one_node_all_rounds) == 0:
            logger.error(
                f'Couldn\'t find information about "slot_full_rank" for node {node}. Skipping plot!'
            )
            return
        if len(data_decoded_one_node_all_rounds) == 0:
            logger.error(
                f'Couldn\'t find information about "decoded" for node {node}. Skipping plot!'
            )
            return

        full_rank_per_node_all_rounds.append(full_rank_one_node_all_rounds)
        data_decoded_per_node_all_rounds.append(data_decoded_one_node_all_rounds)
        slot_off_per_node_all_rounds.append(slot_off_one_node_all_rounds)

    reliability = [
        round(
            sum(node)
            / (len(node) * mlp.exp_config["MX_GENERATION_SIZE"])
            * 100,
            1,
        )
        for node in data_decoded_per_node_all_rounds
    ]

    # plt.subplots(1, 1, figsize=(0.8 * mlp.exp_config['MX_NUM_NODES'], 5))
    plt.figure(figsize=(0.8 * mlp.exp_config["MX_NUM_NODES"], 5))

    if len(slot_off_per_node_all_rounds[0]):
        plt.subplot(2, 1, 2)
        plt.violinplot(
            slot_off_per_node_all_rounds,
            showmeans=True,
            showmedians=False,
            showextrema=False,
        )
        percs = []
        for n in slot_off_per_node_all_rounds:
            p = np.percentile(n, [5, 95])  # , axis=1)
            percs.append(p)
        for i, p in enumerate(percs, 1):
            plt.hlines(p[0], xmin=i - 0.1, xmax=i + 0.1)
            plt.hlines(p[1], xmin=i - 0.1, xmax=i + 0.1)
        bot, top = plt.ylim()
        plt.ylim([bot - 24, top])
        plt.xticks(
            ticks=range(1, len(results) + 1),
            labels=sorted(results.keys()),
        )
        plt.grid(axis="y")
        plt.ylabel("Turn-off Slot")
        plt.subplot(2, 1, 1)

    plt.violinplot(
        full_rank_per_node_all_rounds,
        showmeans=True,
        showmedians=False,
        showextrema=False,
    )

    percs = []
    for n in full_rank_per_node_all_rounds:
        p = np.percentile(n, [5, 95])  # , axis=1)
        percs.append(p)

    for i, p in enumerate(percs, 1):
        plt.hlines(p[0], xmin=i - 0.1, xmax=i + 0.1)
        plt.hlines(p[1], xmin=i - 0.1, xmax=i + 0.1)

    bot, top = plt.ylim()
    plt.ylim([bot - 24, top])
    plt.xticks(
        ticks=range(1, len(results) + 1),
        labels=sorted(results.keys()),
    )

    for i, v in enumerate(reliability, 1):
        if v >= 0:
            plt.text(i, bot - 12, f"{v}%", horizontalalignment="center")

    plt.xlabel("Logical Node ID")
    plt.ylabel("Full Rank Slot")
    plt.title(
        f'{mlp.exp_config["MX_PHY_NAME"]}, time needed for full rank over {num_rounds} rounds\n({mlp.basepath.name})'
    )
    # plt.legend(labels=[str(i) for i in range(1,21)], bbox_to_anchor=(1.05, 1))
    # plt.show()
    plt.gcf().savefig(outputfile, bbox_inches="tight")
    plt.close()
    return outputfile


def info_vector_usage(mlp, results, force=False):

    outputfile = mlp.plotPath / "info_vector_usage.pdf"
    if outputfile.exists() and not force:
        logger.info(
            f"{outputfile} already exists. Delete the file to redo the plot."
        )
        return outputfile

    type_map = {
        0x00 : 0,   # row map
        0x02 : 1,   # row request
        0x03 : 1,   # column request
        0x04 : 3,   # weak-zero map
        0x05 : 4,   # aggregate
        0x80 : 2,   # full-rank map
        0x81 : 2,   # full-rank map + radio off
        0x82 : 2,   # full-rank ack map
        0x83 : 2,   # full-rank ack map + radio off
        0x84 : 3,   # weak-zero map
        0x85 : 4,   # aggregate
    }
    legend = [
        "row map",
        "request",
        "full-rank map",
        "weak-zero map",
        "aggregate"
    ]

    # rounds completed by all nodes
    rounds = set.intersection(*[set(x.keys()) for x in list(results.values())])

    x = range(1, 1 + mlp.exp_config["MX_ROUND_LENGTH"])
    y = np.zeros((1 + max(type_map.values()), len(x)), dtype=np.uint)

    hist = [[] for _ in range(y.shape[0])]

    for node in results:
        for round in rounds:
            if "info_type_per_slot" not in results[node][round]:
                logger.warning(
                    f"unknown info vector types in node {node} round {round}"
                )
                continue
            for slot, type in results[node][round]["info_type_per_slot"].items():
                if type in type_map:
                    y[type_map[type], slot - x[0]] += 1
                    hist[type_map[type]].append(slot)

    plt.figure()
#    plt.figure(figsize=(15, 0.3 * mlp.exp_config["MX_NUM_NODES"]))

#     plt.subplot(2, 1, 1)
#     for i in range(y.shape[0]):
#         plt.plot(x, y[i])
#     plt.subplot(2, 1, 2)

    # convert hist to sequence of arrays
    for k in range(len(hist)):
        hist[k] = np.array(hist[k], dtype=np.float32)

    # create dummy plot to get bins, round intervals, make last bin half-open
    bins = plt.hist(hist, bins=40)[1]
    bins = bins.astype(int)
    bins[-1] += 1

    # compute weights such that plot shows packets per slot instead of packets per slot interval
    weights = []
    for k in range(len(hist)):
        weights.append(np.ones(hist[k].shape) / len(rounds))
        for i in range(len(bins) - 1):
            weights[k][(hist[k] >= bins[i]) & (hist[k] < bins[i+1])] /= bins[i+1] - bins[i]

#     print(bins)
#     print(hist)
#     print(weights)

    # create weighted plot
    plt.clf()
    plt.hist(hist, bins, weights=weights, histtype="stepfilled", stacked=True, label=legend)
    plt.xlim(0, x[-1])
    plt.grid()
    plt.legend()
    plt.xlabel("slot")
    plt.ylabel("transmitted packets per slot")

#    plt.yticks(ticks=np.arange(len(results.keys())), labels=sorted(results.keys()))
#    plt.ylim(len(results.keys()) - 0.5, -0.5)
#    plt.title(
#        f'{mlp.exp_config["MX_PHY_NAME"]}, {funcName} ranks per slot over {num_rounds} rounds\n({mlp.basepath.name})'
#    )

    plt.tight_layout()
    plt.gcf().savefig(outputfile, bbox_inches="tight")
    plt.close()
    return outputfile


def aggregate_progress(mlp, results, force=False):

    outputfile = mlp.plotPath / "aggregate_progress.pdf"
    if outputfile.exists() and not force:
        logger.info(
            f"{outputfile} already exists. Delete the file to redo the plot."
        )
        return outputfile

    # rounds completed by all nodes
    rounds = set.intersection(*[set(x.keys()) for x in list(results.values())])

    # select round to plot
    round = min(rounds)

    for node in results:
        if "agg" not in results[node][round] or \
        "agg_per_slot" not in results[node][round]:
            logger.debug(f"no aggregate data, skipping {outputfile}")
            return None

    # phase_mul > 1 can be used to increase separation between phases in plot's y axis
    phase_mul = 1.1

    MX_NUM_NODES = mlp.exp_config["MX_NUM_NODES"] #len(results)
    MX_NUM_SLOTS = mlp.exp_config["MX_ROUND_LENGTH"]

    x = range(0, MX_NUM_SLOTS)
    progress = np.zeros((len(x), MX_NUM_NODES))
    prop_per_node = -np.ones(progress.shape, dtype=np.int16)

    legend = [f"P{i}" for i in results.keys()]
    legend += [f"P[{i+1}?]" for i in range(len(legend), MX_NUM_NODES)]

    prios_complete = [False] * MX_NUM_NODES

    for i,node in enumerate(results):
        agg_data = results[node][round]["agg_per_slot"]
        slot_lost = results[node][round]["agg"]["slot_proposer_lost"]
        slot_lost = np.inf if slot_lost == 0 else slot_lost
        for slot in sorted(agg_data):
            data = agg_data[slot]
            proposal = data["proposal"]
            if 0 == data["phase"]:
                progress[slot:, proposal] = data["progress"] / MX_NUM_NODES
                prop_per_node[slot:, i] = i
            else:
                if proposal >= MX_NUM_NODES:
                    proposal %= MX_NUM_NODES
                    prios_complete[proposal] = True
                p = data["phase"] * phase_mul + data["progress"] / MX_NUM_NODES
                if 1:   # show max. progress in the network
                    for k in range(slot, MX_NUM_SLOTS):
                        if progress[k, proposal] >= p:
                            break
                        progress[k, proposal] = p
                else:   # show progress seen by proposer
                    if proposal == i:
                        progress[slot:, proposal] = p
                if slot >= slot_lost:
                    prop_per_node[slot:, i] = data["min_proposal"] % MX_NUM_NODES

    nodes_per_prop = np.zeros(prop_per_node.shape, dtype=np.uint16)
    for slot in range(prop_per_node.shape[0]):
        for node in range(prop_per_node.shape[1]):
            if prop_per_node[slot, node] >= 0:
                nodes_per_prop[slot, prop_per_node[slot, node]] += 1

    # cut plots of losing proposals
    for i,node in enumerate(results):
        slot = results[node][round]["agg"]["slot_proposer_lost"]
        if slot > 0:
            progress[slot:, i] = np.nan
#     # alternative variant, less reliable (always prefer slot_proposer_lost if available)
#     for i,node in enumerate(results):
#         agg_data = results[node][round]["agg_per_slot"]
#         for slot in sorted(agg_data):
#             if np.isnan(progress[slot, i]):
#                 break
#             data = agg_data[slot]
#             pack_phase = data["phase"]
#             node_phase = int(progress[slot, i] / phase_mul)
#             if node_phase > pack_phase:
#                 continue
#             pack_prop = data["proposal"]
#             node_prop = i + prios_complete[i] * MX_NUM_NODES
#             #print(f"{pack_prop} <> {node_prop}")
#             if node_phase < pack_phase or \
#                 node_phase == 1 and data["min_proposal"] > node_prop or \
#                 node_phase == 2 and data["max_accepted_proposal"] > node_prop:
#                 progress[slot:, i] = np.nan
#                 break

    for i in [i for i, x in enumerate(prios_complete) if x]:
        legend[i] += " (*)"

    plt.figure()

    plt.subplot(2, 1, 1)
    hplot = plt.plot(x, progress)
    plt.ylim(0, 3 * phase_mul)
    plt.yticks(
        phase_mul * np.array([0, 1, 2]),
        ["Collect 0%", "Prepare 0%", "Accept 0%"],
        minor=False)
    plt.yticks(
        phase_mul * np.array([0, 1, 1, 2, 2]) + [1, 0.5, 1, 0.5, 1],
        ["", "50%", "", "50%", "100%"],
        minor=True)
    plt.grid(axis="y", color="k")
    plt.grid(which="minor", linestyle="--")
    plt.xlabel("slot")
    plt.ylabel("progress")
    plt.legend(legend) #, loc="upper left")

    for i,node in enumerate(results):
        plt.axvline(x = results[node][round]["agg"]["slot_value_learned"],
            linestyle="--", color=plt.get(hplot[i], "color"))

    plt.subplot(2, 1, 2)
    plt.plot(x, nodes_per_prop)
    plt.grid(axis="y")
    plt.ylabel("#nodes tracking proposal")

    plt.tight_layout()
    plt.gcf().savefig(outputfile, bbox_inches="tight")
    plt.close()
    return outputfile


def create_config_pdf(mlp):
    outputfile = mlp.plotPath / "config.pdf"

    pdf = FPDF(format=(400, 300))
    pdf.add_page()
    pdf.set_font("Courier", "", 12)
    key_width = 25

    for k, v in mlp.exp_config.items():
        # NODE_MAPPING usually spans multiple lines and needs to be split into chunks
        if k == "NODE_MAPPING":
            nodemap = list(v)
            for i in range(0, len(nodemap), 10):
                # s = str(nodemap[i:i+10]).strip('[] ')
                s = ""
                for node in nodemap[i : i + 10]:
                    s += f"{str(node):<10},"
                if i == 0:
                    pdf.cell(0, 5, f"{k:<{key_width}}: {s}", ln=1)
                else:
                    pdf.cell(0, 5, f'{"":<{key_width}}  {s}', ln=1)
        else:
            pdf.cell(0, 5, f"{k:<{key_width}}: {v}", ln=1)

    pdf.output(outputfile, "F")
    return outputfile


def create_overview(mlp, plotFiles):
    pdfMerger = PyPDF2.PdfMerger()
    for pdf in plotFiles:
        pdfMerger.append(PyPDF2.PdfReader(str(pdf)))

    with open(mlp.plotPath / "overview.pdf", "wb") as f:
        pdfMerger.write(f)


def create_experiment_plots(path, force=False):
    mlp = MixerLogParser(path)

    file_handler = logging.FileHandler(
        mlp.plotPath / "stats.log", mode="w", encoding="utf-8"
    )
    formatter = logging.Formatter(
        fmt="%(filename)-15.15s:%(lineno)-5d %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    results = extract_infos(mlp.log_formatted, mlp.exp_config)

    # print information about rounds
    min_rounds = min([(len(rnds), node) for node, rnds in results.items()])
    logger.info(
        f"Min number of rounds completed by node {min_rounds[1]}: {min_rounds[0]}"
    )
    max_rounds = max([(len(rnds), node) for node, rnds in results.items()])
    logger.info(
        f"Max number of rounds completed by node {max_rounds[1]}: {max_rounds[0]}"
    )

    common_rnds = list(set.intersection(*[set(x.keys()) for x in list(results.values())]))
#    rnds = [list(rnds.keys()) for rnds in results.values()]
#    common_rnds = rnds[0]
#    for rnd in rnds[1:]:
#        common_rnds = list(set(common_rnds) & set(rnd))
    logger.info(f"{len(common_rnds)} rounds completed by all nodes")

    # strip first and last round
#    common_rnds = common_rnds[0:-1]
    common_rnds = common_rnds[1:-1]
    logger.info("Stripping first and last round")

    results_tmp = copy.deepcopy(results)
    for node, rnds in results_tmp.items():
        for rnd_id, rnd in rnds.items():
            if rnd_id not in common_rnds:
                del results[node][rnd_id]

    plotFiles = []

    p = create_config_pdf(mlp)
    if p and p.exists():
        plotFiles.append(p)

    p = slots_node_rank_heatmap_func(
        mlp, results, lambda x: np.mean(x), "mean", force
    )
    if p and p.exists():
        plotFiles.append(p)

    p = slots_node_rank_heatmap_func(
        mlp, results, lambda x: min(x), "min", force
    )
    if p and p.exists():
        plotFiles.append(p)

    p = slots_node_rank_heatmap_func(
        mlp, results, lambda x: max(x), "max", force
    )
    if p and p.exists():
        plotFiles.append(p)

    p = node_fullRankSlot_reliability_violin(mlp, results, force)
    if p and p.exists():
        plotFiles.append(p)

    p = slots_rank_linePerNode_func(
        mlp, results, lambda x: np.mean(x), "mean", force
    )
    if p and p.exists():
        plotFiles.append(p)

    p = node_discoveryNeighbors_violin(mlp, results, force)
    if p and p.exists():
        plotFiles.append(p)

    p = info_vector_usage(mlp, results, force)
    if p and p.exists():
        plotFiles.append(p)

    p = aggregate_progress(mlp, results, force)
    if p and p.exists():
        plotFiles.append(p)

    if G.sync_stats:
        p = sync_groups_time_line_all(mlp, results, force)
        if p and p.exists():
            plotFiles.append(p)

        p = sync_groups_all_violin(mlp, results, force)
        if p and p.exists():
            plotFiles.append(p)

        p = sync_groups_time_line_single(mlp, results, force)
        if p and p.exists():
            plotFiles.append(p)

        p = sync_leader_distribution(mlp, results, force)
        if p and p.exists():
            plotFiles.append(p)

        sync_stats(mlp, results, force)

    # exclude rounds plot from overview
    # slots_node_rank_heatmap_rounds(mlp, results, (0,19), force)
    # slots_node_rank_heatmap_rounds(mlp, results, (105,119), force)

    create_overview(mlp, plotFiles)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to log directory")
    parser.add_argument(
        "--lvl", help="specifies log level", choices=["INFO", "DEBUG"]
    )
    parser.add_argument(
        "--all",
        help="path contains multiple experiments that should be all evaluated",
        action="store_true",
    )
    parser.add_argument(
        "--force",
        help="force plotting even if the plot already exists",
        action="store_true",
    )
    args = parser.parse_args()

    if not args.lvl:
        logger.setLevel(logging.INFO)
    elif args.lvl == "INFO":
        logger.setLevel(logging.INFO)
    elif args.lvl == "DEBUG":
        logger.setLevel(logging.DEBUG)
    else:
        logger.critical("Unknown log level")
        sys.exit()

    if args.all:
        for path in Path(args.path).iterdir():
            create_experiment_plots(path, args.force)
    else:
        create_experiment_plots(Path(args.path), args.force)


if __name__ == "__main__":
    main()
