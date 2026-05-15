import random
import numpy as np

class GlobalVars:

	# class variables shared by all instances
	nodes = {}

	config = {
        'label': 'config 1',
        'fieldSize': 2,
        'payloadDistribution': lambda : list(range(1, len(nodes) + 1)),
        'fInitiator': lambda r, n : 1,
        'fTimeout': lambda n : random.randint(2,6),
        'historyWindowLength': [round(x) for x in np.array([3, 1]) * 10],
        'exchangeTriggerSparsity': lambda : len(nodes),
        'immediateElimination': False,
        'fastTxUpdate': {
            'allowedUpdates': 1,
            'mulOnRx': True,
            'mulOnRequest': True,
            'mulOnOwn': True,
        },
        'smartShutdown': True,
        'recursiveNeighborhood': 1,
        'txPacket': {
            'includeOwn': 3, # -1 = always, 0 = unenforced, x>0 = until x times received (= acked)
            'fAgeToP': lambda a : 0.5 + (1 - 0.5) * 2 ** (-0.5 * a),
            'emptyPacketStrategy': 'own' # what to insert: 'own' = own, 'first' = first found; 'random' = random one
        },
        'fTxCurve': lambda a, d, n=0, dn=0 : 1 / (d + 1) + d / (d + 1) * 2 ** (-0.5 * a),
        'coordinatedSlotting': {
            'on': True,
            'fpOwn': lambda p, d, n, h : 1 - h,
            'fpForeign': lambda p, d, n, h : h / (d + 1),
            'fpInit': lambda p, n : 1 / n,
        },
        'request': {
            'mode': 'column,pivot',
            'columnSearchMode': 'pivot', # pivot or all
            'rxSnoop': True,
            'fTxColumnYesNo': lambda a, r, n : a > (n - r),
            'fTxPivotYesNo': lambda a, r, n : a > (n - r),
            # 'fTxSelect': lambda m : random.random() < np.count_nonzero(m) / len(m), # octave length(m) returns 1 if scalar, number of elements if vector or number of elements of the biggest dimension if matrix
            'fpHelpless': lambda a, d, n : 1 / n,
            'numRounds': 0,
            'numSlots': 0,
            'payloadSize': 0
        }
        # traceMap
        # traceFile
        # fLineHeader
        # simulationMode
        # rounds
    }

    testcase = {
        'paramStrings': '',
        'simulationMode': 'default',
        'testbeds':
        'configSource':
        'config':
        'logLevel':
        'logDetailsMode':
    }
