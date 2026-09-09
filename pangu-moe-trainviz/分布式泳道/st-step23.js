/* Extracted ST step 23; see export-st-fixture.cjs and spec for provenance. */
(function(root) { const data = {
  "source": "https://github.com/ByteDance-Seed/StragglerAnalysis/blob/main/data/timeline-ST.json.gz",
  "configSource": "https://github.com/ByteDance-Seed/StragglerAnalysis/blob/main/data/meta-ST.yaml",
  "sourceSha256": "17ebc492623ec67fefc4c56152732b64f24020535452078309616a2aea57686c",
  "step": 23,
  "topology": {
    "dp": 2,
    "pp": 4,
    "tp": 1,
    "dsp": 1,
    "vpp": 1,
    "world": 8
  },
  "sourceTimeUnit": "us",
  "structureMapping": null,
  "events": [
    {
      "name": "backward-compute",
      "cat": "backward-compute",
      "pid": 0,
      "tid": 0,
      "ts": 848671.6747283936,
      "dur": 138679.68320846558,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 0,
        "seq_id": 0
      },
      "sourceIndex": 0
    },
    {
      "name": "backward-compute",
      "cat": "backward-compute",
      "pid": 0,
      "tid": 0,
      "ts": 1182869.9111938477,
      "dur": 138663.649559021,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 1,
        "seq_id": 1
      },
      "sourceIndex": 1
    },
    {
      "name": "backward-compute",
      "cat": "backward-compute",
      "pid": 0,
      "tid": 0,
      "ts": 1515103.3401489258,
      "dur": 138442.42691993713,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 2,
        "seq_id": 2
      },
      "sourceIndex": 2
    },
    {
      "name": "backward-compute",
      "cat": "backward-compute",
      "pid": 0,
      "tid": 0,
      "ts": 1846184.492111206,
      "dur": 138901.53169631958,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 3,
        "seq_id": 3
      },
      "sourceIndex": 3
    },
    {
      "name": "backward-recv",
      "cat": "backward-recv",
      "pid": 0,
      "tid": 1,
      "ts": 847876.4295578003,
      "dur": 802.0401000976562,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 4
    },
    {
      "name": "backward-recv",
      "cat": "backward-recv",
      "pid": 0,
      "tid": 1,
      "ts": 1182410.329580307,
      "dur": 465.15464782714844,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 1
      },
      "sourceIndex": 5
    },
    {
      "name": "backward-recv",
      "cat": "backward-recv",
      "pid": 0,
      "tid": 1,
      "ts": 1514642.8495645523,
      "dur": 467.53883361816406,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 2
      },
      "sourceIndex": 6
    },
    {
      "name": "backward-recv",
      "cat": "backward-recv",
      "pid": 0,
      "tid": 1,
      "ts": 1845723.6588001251,
      "dur": 457.2868347167969,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 3
      },
      "sourceIndex": 7
    },
    {
      "name": "embedding-grads-all-reduce",
      "cat": "embedding-grads-all-reduce",
      "pid": 0,
      "tid": 3,
      "ts": 1986141.2048339844,
      "dur": 266300.4398345947,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 8
    },
    {
      "name": "forward-compute",
      "cat": "forward-compute",
      "pid": 0,
      "tid": 0,
      "ts": 48943.519592285156,
      "dur": 63458.97912979126,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 0,
        "seq_id": 0
      },
      "sourceIndex": 9
    },
    {
      "name": "forward-compute",
      "cat": "forward-compute",
      "pid": 0,
      "tid": 0,
      "ts": 112787.00828552246,
      "dur": 64271.099865436554,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 1,
        "seq_id": 1
      },
      "sourceIndex": 10
    },
    {
      "name": "forward-compute",
      "cat": "forward-compute",
      "pid": 0,
      "tid": 0,
      "ts": 177447.08061218262,
      "dur": 65395.7724571228,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 2,
        "seq_id": 2
      },
      "sourceIndex": 11
    },
    {
      "name": "forward-compute",
      "cat": "forward-compute",
      "pid": 0,
      "tid": 0,
      "ts": 243224.1439819336,
      "dur": 64877.890050411224,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 3,
        "seq_id": 3
      },
      "sourceIndex": 12
    },
    {
      "name": "forward-send",
      "cat": "forward-send",
      "pid": 0,
      "tid": 1,
      "ts": 112404.0688155219,
      "dur": 380.0392150878906,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 13
    },
    {
      "name": "forward-send",
      "cat": "forward-send",
      "pid": 0,
      "tid": 1,
      "ts": 177064.2589253839,
      "dur": 388.14544677734375,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 1
      },
      "sourceIndex": 14
    },
    {
      "name": "forward-send",
      "cat": "forward-send",
      "pid": 0,
      "tid": 1,
      "ts": 242845.40602820925,
      "dur": 385.52284240722656,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 2
      },
      "sourceIndex": 15
    },
    {
      "name": "forward-send",
      "cat": "forward-send",
      "pid": 0,
      "tid": 1,
      "ts": 847876.4295578003,
      "dur": 802.0401000976562,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 3
      },
      "sourceIndex": 16
    },
    {
      "name": "gc",
      "cat": "gc",
      "pid": 0,
      "tid": 0,
      "ts": 1985085.9642028809,
      "dur": 0,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": 0,
        "seq_id": 0
      },
      "sourceIndex": 17
    },
    {
      "name": "grads-reduce-scatter",
      "cat": "grads-reduce-scatter",
      "pid": 0,
      "tid": 3,
      "ts": 2269191.287457943,
      "dur": 55544.137954711914,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 18
    },
    {
      "name": "layernorm-grads-all-reduce",
      "cat": "layernorm-grads-all-reduce",
      "pid": 0,
      "tid": 3,
      "ts": 1986074.9244689941,
      "dur": 46.30399780580774,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 19
    },
    {
      "name": "optimizer",
      "cat": "optimizer",
      "pid": 0,
      "tid": 0,
      "ts": 2339182.3768615723,
      "dur": 42362.45155334473,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 20
    },
    {
      "name": "optimizer-clip-main-grad",
      "cat": "optimizer-clip-main-grad",
      "pid": 0,
      "tid": 3,
      "ts": 2324741.944670677,
      "dur": 14440.536499023438,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 21
    },
    {
      "name": "params-all-gather",
      "cat": "params-all-gather",
      "pid": 0,
      "tid": 3,
      "ts": 5341.999232769012,
      "dur": 27061.22398376465,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 22
    },
    {
      "name": "backward-compute",
      "cat": "backward-compute",
      "pid": 1,
      "tid": 5,
      "ts": 831830.0247192383,
      "dur": 133070.49870491028,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 0,
        "seq_id": 0
      },
      "sourceIndex": 23
    },
    {
      "name": "backward-compute",
      "cat": "backward-compute",
      "pid": 1,
      "tid": 5,
      "ts": 1174157.1426391602,
      "dur": 136741.5338754654,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 1,
        "seq_id": 1
      },
      "sourceIndex": 24
    },
    {
      "name": "backward-compute",
      "cat": "backward-compute",
      "pid": 1,
      "tid": 5,
      "ts": 1492148.6377716064,
      "dur": 132783.9344739914,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 2,
        "seq_id": 2
      },
      "sourceIndex": 25
    },
    {
      "name": "backward-compute",
      "cat": "backward-compute",
      "pid": 1,
      "tid": 5,
      "ts": 1825491.4283752441,
      "dur": 133498.59416484833,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 3,
        "seq_id": 3
      },
      "sourceIndex": 26
    },
    {
      "name": "backward-recv",
      "cat": "backward-recv",
      "pid": 1,
      "tid": 6,
      "ts": 831058.9790344238,
      "dur": 779.1519165039062,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 27
    },
    {
      "name": "backward-recv",
      "cat": "backward-recv",
      "pid": 1,
      "tid": 6,
      "ts": 1173701.5396356583,
      "dur": 451.08795166015625,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 1
      },
      "sourceIndex": 28
    },
    {
      "name": "backward-recv",
      "cat": "backward-recv",
      "pid": 1,
      "tid": 6,
      "ts": 1491699.6210813522,
      "dur": 447.5116729736328,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 2
      },
      "sourceIndex": 29
    },
    {
      "name": "backward-recv",
      "cat": "backward-recv",
      "pid": 1,
      "tid": 6,
      "ts": 1825045.4366207123,
      "dur": 440.5975341796875,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 3
      },
      "sourceIndex": 30
    },
    {
      "name": "embedding-grads-all-reduce",
      "cat": "embedding-grads-all-reduce",
      "pid": 1,
      "tid": 8,
      "ts": 1960034.1618061066,
      "dur": 289930.8204650879,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 31
    },
    {
      "name": "forward-compute",
      "cat": "forward-compute",
      "pid": 1,
      "tid": 5,
      "ts": 48968.79196166992,
      "dur": 62002.78550386429,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 0,
        "seq_id": 0
      },
      "sourceIndex": 32
    },
    {
      "name": "forward-compute",
      "cat": "forward-compute",
      "pid": 1,
      "tid": 5,
      "ts": 111347.43690490723,
      "dur": 62987.67775297165,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 1,
        "seq_id": 1
      },
      "sourceIndex": 33
    },
    {
      "name": "forward-compute",
      "cat": "forward-compute",
      "pid": 1,
      "tid": 5,
      "ts": 174718.37997436523,
      "dur": 62109.98445749283,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 2,
        "seq_id": 2
      },
      "sourceIndex": 34
    },
    {
      "name": "forward-compute",
      "cat": "forward-compute",
      "pid": 1,
      "tid": 5,
      "ts": 238893.98574829102,
      "dur": 62271.200120449066,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 3,
        "seq_id": 3
      },
      "sourceIndex": 35
    },
    {
      "name": "forward-send",
      "cat": "forward-send",
      "pid": 1,
      "tid": 6,
      "ts": 110972.31961321086,
      "dur": 370.7408905029297,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 36
    },
    {
      "name": "forward-send",
      "cat": "forward-send",
      "pid": 1,
      "tid": 6,
      "ts": 174339.41559283994,
      "dur": 375.50926208496094,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 1
      },
      "sourceIndex": 37
    },
    {
      "name": "forward-send",
      "cat": "forward-send",
      "pid": 1,
      "tid": 6,
      "ts": 238605.28599470854,
      "dur": 285.14862060546875,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 2
      },
      "sourceIndex": 38
    },
    {
      "name": "forward-send",
      "cat": "forward-send",
      "pid": 1,
      "tid": 6,
      "ts": 831058.9790344238,
      "dur": 779.1519165039062,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 3
      },
      "sourceIndex": 39
    },
    {
      "name": "gc",
      "cat": "gc",
      "pid": 1,
      "tid": 5,
      "ts": 1958990.0970458984,
      "dur": 0,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": 0,
        "seq_id": 0
      },
      "sourceIndex": 40
    },
    {
      "name": "grads-reduce-scatter",
      "cat": "grads-reduce-scatter",
      "pid": 1,
      "tid": 8,
      "ts": 2269191.190600395,
      "dur": 55552.72102355957,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 41
    },
    {
      "name": "layernorm-grads-all-reduce",
      "cat": "layernorm-grads-all-reduce",
      "pid": 1,
      "tid": 8,
      "ts": 1959975.4810333252,
      "dur": 38.623998989351094,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 42
    },
    {
      "name": "optimizer",
      "cat": "optimizer",
      "pid": 1,
      "tid": 5,
      "ts": 2339171.886444092,
      "dur": 42233.943939208984,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 43
    },
    {
      "name": "optimizer-clip-main-grad",
      "cat": "optimizer-clip-main-grad",
      "pid": 1,
      "tid": 8,
      "ts": 2324741.804972291,
      "dur": 14430.046081542969,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 44
    },
    {
      "name": "params-all-gather",
      "cat": "params-all-gather",
      "pid": 1,
      "tid": 8,
      "ts": 5341.9433534145355,
      "dur": 27061.46240234375,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 45
    },
    {
      "name": "backward-compute",
      "cat": "backward-compute",
      "pid": 0,
      "tid": 10,
      "ts": 710927.9632568359,
      "dur": 136945.27745246887,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 0,
        "seq_id": 0
      },
      "sourceIndex": 46
    },
    {
      "name": "backward-compute",
      "cat": "backward-compute",
      "pid": 0,
      "tid": 10,
      "ts": 1044324.8748779297,
      "dur": 138077.21436023712,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 1,
        "seq_id": 1
      },
      "sourceIndex": 47
    },
    {
      "name": "backward-compute",
      "cat": "backward-compute",
      "pid": 0,
      "tid": 10,
      "ts": 1376534.2235565186,
      "dur": 138104.70700263977,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 2,
        "seq_id": 2
      },
      "sourceIndex": 48
    },
    {
      "name": "backward-compute",
      "cat": "backward-compute",
      "pid": 0,
      "tid": 10,
      "ts": 1709587.3355865479,
      "dur": 136132.27009773254,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 3,
        "seq_id": 3
      },
      "sourceIndex": 49
    },
    {
      "name": "backward-recv",
      "cat": "backward-recv",
      "pid": 0,
      "tid": 11,
      "ts": 710107.684135437,
      "dur": 829.6966552734375,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 50
    },
    {
      "name": "backward-recv",
      "cat": "backward-recv",
      "pid": 0,
      "tid": 11,
      "ts": 1043502.1966695786,
      "dur": 831.1271667480469,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 1
      },
      "sourceIndex": 51
    },
    {
      "name": "backward-recv",
      "cat": "backward-recv",
      "pid": 0,
      "tid": 11,
      "ts": 1376048.6990213394,
      "dur": 479.2213439941406,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 2
      },
      "sourceIndex": 52
    },
    {
      "name": "backward-recv",
      "cat": "backward-recv",
      "pid": 0,
      "tid": 11,
      "ts": 1709107.1605682373,
      "dur": 486.61231994628906,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 3
      },
      "sourceIndex": 53
    },
    {
      "name": "backward-send",
      "cat": "backward-send",
      "pid": 0,
      "tid": 11,
      "ts": 847876.6629705206,
      "dur": 772.2377777099609,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 54
    },
    {
      "name": "backward-send",
      "cat": "backward-send",
      "pid": 0,
      "tid": 11,
      "ts": 1182410.3141843807,
      "dur": 385.04600524902344,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 1
      },
      "sourceIndex": 55
    },
    {
      "name": "backward-send",
      "cat": "backward-send",
      "pid": 0,
      "tid": 11,
      "ts": 1514642.9775340948,
      "dur": 394.3443298339844,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 2
      },
      "sourceIndex": 56
    },
    {
      "name": "backward-send",
      "cat": "backward-send",
      "pid": 0,
      "tid": 11,
      "ts": 1845723.534177523,
      "dur": 395.77484130859375,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 3
      },
      "sourceIndex": 57
    },
    {
      "name": "forward-compute",
      "cat": "forward-compute",
      "pid": 0,
      "tid": 10,
      "ts": 112860.44120788574,
      "dur": 63476.95738077164,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 0,
        "seq_id": 0
      },
      "sourceIndex": 58
    },
    {
      "name": "forward-compute",
      "cat": "forward-compute",
      "pid": 0,
      "tid": 10,
      "ts": 177524.32823181152,
      "dur": 64336.70222759247,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 1,
        "seq_id": 1
      },
      "sourceIndex": 59
    },
    {
      "name": "forward-compute",
      "cat": "forward-compute",
      "pid": 0,
      "tid": 10,
      "ts": 243309.02099609375,
      "dur": 64400.508999824524,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 2,
        "seq_id": 2
      },
      "sourceIndex": 60
    },
    {
      "name": "forward-compute",
      "cat": "forward-compute",
      "pid": 0,
      "tid": 10,
      "ts": 848646.1639404297,
      "dur": 65318.748354911804,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 3,
        "seq_id": 3
      },
      "sourceIndex": 61
    },
    {
      "name": "forward-recv",
      "cat": "forward-recv",
      "pid": 0,
      "tid": 11,
      "ts": 112404.16765213013,
      "dur": 450.6111145019531,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 62
    },
    {
      "name": "forward-recv",
      "cat": "forward-recv",
      "pid": 0,
      "tid": 11,
      "ts": 177064.1456823796,
      "dur": 454.6642303466797,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 1
      },
      "sourceIndex": 63
    },
    {
      "name": "forward-recv",
      "cat": "forward-recv",
      "pid": 0,
      "tid": 11,
      "ts": 242845.32817546278,
      "dur": 457.52525329589844,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 2
      },
      "sourceIndex": 64
    },
    {
      "name": "forward-recv",
      "cat": "forward-recv",
      "pid": 0,
      "tid": 11,
      "ts": 847876.6629705206,
      "dur": 772.2377777099609,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 3
      },
      "sourceIndex": 65
    },
    {
      "name": "forward-send",
      "cat": "forward-send",
      "pid": 0,
      "tid": 11,
      "ts": 176340.9304257948,
      "dur": 369.07196044921875,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 66
    },
    {
      "name": "forward-send",
      "cat": "forward-send",
      "pid": 0,
      "tid": 11,
      "ts": 241863.6754155159,
      "dur": 361.2041473388672,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 1
      },
      "sourceIndex": 67
    },
    {
      "name": "forward-send",
      "cat": "forward-send",
      "pid": 0,
      "tid": 11,
      "ts": 710107.684135437,
      "dur": 829.6966552734375,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 2
      },
      "sourceIndex": 68
    },
    {
      "name": "forward-send",
      "cat": "forward-send",
      "pid": 0,
      "tid": 11,
      "ts": 1043502.1966695786,
      "dur": 831.1271667480469,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 3
      },
      "sourceIndex": 69
    },
    {
      "name": "gc",
      "cat": "gc",
      "pid": 0,
      "tid": 10,
      "ts": 1845719.575881958,
      "dur": 0,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": 0,
        "seq_id": 0
      },
      "sourceIndex": 70
    },
    {
      "name": "grads-reduce-scatter",
      "cat": "grads-reduce-scatter",
      "pid": 0,
      "tid": 13,
      "ts": 1862157.0393443108,
      "dur": 50011.396408081055,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 71
    },
    {
      "name": "layernorm-grads-all-reduce",
      "cat": "layernorm-grads-all-reduce",
      "pid": 0,
      "tid": 13,
      "ts": 1846997.0226287842,
      "dur": 35.42400008882396,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 72
    },
    {
      "name": "optimizer",
      "cat": "optimizer",
      "pid": 0,
      "tid": 10,
      "ts": 2338346.242904663,
      "dur": 37961.721420288086,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 73
    },
    {
      "name": "optimizer-clip-main-grad",
      "cat": "optimizer-clip-main-grad",
      "pid": 0,
      "tid": 13,
      "ts": 2324741.7509555817,
      "dur": 13604.402542114258,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 74
    },
    {
      "name": "params-all-gather",
      "cat": "params-all-gather",
      "pid": 0,
      "tid": 13,
      "ts": 44.152140617370605,
      "dur": 23013.830184936523,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 75
    },
    {
      "name": "backward-compute",
      "cat": "backward-compute",
      "pid": 1,
      "tid": 15,
      "ts": 697516.679763794,
      "dur": 133536.48781776428,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 0,
        "seq_id": 0
      },
      "sourceIndex": 76
    },
    {
      "name": "backward-compute",
      "cat": "backward-compute",
      "pid": 1,
      "tid": 15,
      "ts": 1035682.6782226562,
      "dur": 138017.87793636322,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 1,
        "seq_id": 1
      },
      "sourceIndex": 77
    },
    {
      "name": "backward-compute",
      "cat": "backward-compute",
      "pid": 1,
      "tid": 15,
      "ts": 1359610.5575561523,
      "dur": 132086.78364753723,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 2,
        "seq_id": 2
      },
      "sourceIndex": 78
    },
    {
      "name": "backward-compute",
      "cat": "backward-compute",
      "pid": 1,
      "tid": 15,
      "ts": 1690217.9718017578,
      "dur": 134825.24454593658,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 3,
        "seq_id": 3
      },
      "sourceIndex": 79
    },
    {
      "name": "backward-recv",
      "cat": "backward-recv",
      "pid": 1,
      "tid": 16,
      "ts": 696708.0235481262,
      "dur": 814.9147033691406,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 80
    },
    {
      "name": "backward-recv",
      "cat": "backward-recv",
      "pid": 1,
      "tid": 16,
      "ts": 1034882.3815584183,
      "dur": 796.0796356201172,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 1
      },
      "sourceIndex": 81
    },
    {
      "name": "backward-recv",
      "cat": "backward-recv",
      "pid": 1,
      "tid": 16,
      "ts": 1359154.4330120087,
      "dur": 462.53204345703125,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 2
      },
      "sourceIndex": 82
    },
    {
      "name": "backward-recv",
      "cat": "backward-recv",
      "pid": 1,
      "tid": 16,
      "ts": 1689758.3454847336,
      "dur": 464.6778106689453,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 3
      },
      "sourceIndex": 83
    },
    {
      "name": "backward-send",
      "cat": "backward-send",
      "pid": 1,
      "tid": 16,
      "ts": 831058.9276370592,
      "dur": 761.7473602294922,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 84
    },
    {
      "name": "backward-send",
      "cat": "backward-send",
      "pid": 1,
      "tid": 16,
      "ts": 1173701.5392864123,
      "dur": 388.14544677734375,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 1
      },
      "sourceIndex": 85
    },
    {
      "name": "backward-send",
      "cat": "backward-send",
      "pid": 1,
      "tid": 16,
      "ts": 1491699.5817620773,
      "dur": 392.913818359375,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 2
      },
      "sourceIndex": 86
    },
    {
      "name": "backward-send",
      "cat": "backward-send",
      "pid": 1,
      "tid": 16,
      "ts": 1825045.3941582236,
      "dur": 383.3770751953125,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 3
      },
      "sourceIndex": 87
    },
    {
      "name": "forward-compute",
      "cat": "forward-compute",
      "pid": 1,
      "tid": 15,
      "ts": 111445.90377807617,
      "dur": 62111.712992191315,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 0,
        "seq_id": 0
      },
      "sourceIndex": 88
    },
    {
      "name": "forward-compute",
      "cat": "forward-compute",
      "pid": 1,
      "tid": 15,
      "ts": 174808.74061584473,
      "dur": 63416.35435819626,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 1,
        "seq_id": 1
      },
      "sourceIndex": 89
    },
    {
      "name": "forward-compute",
      "cat": "forward-compute",
      "pid": 1,
      "tid": 15,
      "ts": 239058.4945678711,
      "dur": 62716.22329950333,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 2,
        "seq_id": 2
      },
      "sourceIndex": 90
    },
    {
      "name": "forward-compute",
      "cat": "forward-compute",
      "pid": 1,
      "tid": 15,
      "ts": 831812.858581543,
      "dur": 64263.105392456055,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 3,
        "seq_id": 3
      },
      "sourceIndex": 91
    },
    {
      "name": "forward-recv",
      "cat": "forward-recv",
      "pid": 1,
      "tid": 16,
      "ts": 110972.40447998047,
      "dur": 469.207763671875,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 92
    },
    {
      "name": "forward-recv",
      "cat": "forward-recv",
      "pid": 1,
      "tid": 16,
      "ts": 174339.4230143167,
      "dur": 466.34674072265625,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 1
      },
      "sourceIndex": 93
    },
    {
      "name": "forward-recv",
      "cat": "forward-recv",
      "pid": 1,
      "tid": 16,
      "ts": 238605.18267611042,
      "dur": 448.9421844482422,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 2
      },
      "sourceIndex": 94
    },
    {
      "name": "forward-recv",
      "cat": "forward-recv",
      "pid": 1,
      "tid": 16,
      "ts": 831058.9276370592,
      "dur": 761.7473602294922,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 3
      },
      "sourceIndex": 95
    },
    {
      "name": "forward-send",
      "cat": "forward-send",
      "pid": 1,
      "tid": 16,
      "ts": 173558.77903173678,
      "dur": 371.9329833984375,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 96
    },
    {
      "name": "forward-send",
      "cat": "forward-send",
      "pid": 1,
      "tid": 16,
      "ts": 238226.5285414178,
      "dur": 374.3171691894531,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 1
      },
      "sourceIndex": 97
    },
    {
      "name": "forward-send",
      "cat": "forward-send",
      "pid": 1,
      "tid": 16,
      "ts": 696708.0235481262,
      "dur": 814.9147033691406,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 2
      },
      "sourceIndex": 98
    },
    {
      "name": "forward-send",
      "cat": "forward-send",
      "pid": 1,
      "tid": 16,
      "ts": 1034882.3815584183,
      "dur": 796.0796356201172,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 3
      },
      "sourceIndex": 99
    },
    {
      "name": "gc",
      "cat": "gc",
      "pid": 1,
      "tid": 15,
      "ts": 1825043.2014465332,
      "dur": 0,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": 0,
        "seq_id": 0
      },
      "sourceIndex": 100
    },
    {
      "name": "grads-reduce-scatter",
      "cat": "grads-reduce-scatter",
      "pid": 1,
      "tid": 18,
      "ts": 1862157.1734547615,
      "dur": 50022.125244140625,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 101
    },
    {
      "name": "layernorm-grads-all-reduce",
      "cat": "layernorm-grads-all-reduce",
      "pid": 1,
      "tid": 18,
      "ts": 1826381.4449310303,
      "dur": 38.368001696653664,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 102
    },
    {
      "name": "optimizer",
      "cat": "optimizer",
      "pid": 1,
      "tid": 15,
      "ts": 2338325.023651123,
      "dur": 38145.06530761719,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 103
    },
    {
      "name": "optimizer-clip-main-grad",
      "cat": "optimizer-clip-main-grad",
      "pid": 1,
      "tid": 18,
      "ts": 2324741.870164871,
      "dur": 13583.183288574219,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 104
    },
    {
      "name": "params-all-gather",
      "cat": "params-all-gather",
      "pid": 1,
      "tid": 18,
      "ts": 44.06645894050598,
      "dur": 23015.975952148438,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 105
    },
    {
      "name": "backward-compute",
      "cat": "backward-compute",
      "pid": 0,
      "tid": 20,
      "ts": 572354.0782928467,
      "dur": 137749.92525577545,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 0,
        "seq_id": 0
      },
      "sourceIndex": 106
    },
    {
      "name": "backward-compute",
      "cat": "backward-compute",
      "pid": 0,
      "tid": 20,
      "ts": 905580.2822113037,
      "dur": 137918.3679819107,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 1,
        "seq_id": 1
      },
      "sourceIndex": 107
    },
    {
      "name": "backward-compute",
      "cat": "backward-compute",
      "pid": 0,
      "tid": 20,
      "ts": 1238279.3426513672,
      "dur": 137764.8264169693,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 2,
        "seq_id": 2
      },
      "sourceIndex": 108
    },
    {
      "name": "backward-compute",
      "cat": "backward-compute",
      "pid": 0,
      "tid": 20,
      "ts": 1571290.9698486328,
      "dur": 137814.6857023239,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 3,
        "seq_id": 3
      },
      "sourceIndex": 109
    },
    {
      "name": "backward-recv",
      "cat": "backward-recv",
      "pid": 0,
      "tid": 21,
      "ts": 571644.9022293091,
      "dur": 713.5868072509766,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 110
    },
    {
      "name": "backward-recv",
      "cat": "backward-recv",
      "pid": 0,
      "tid": 21,
      "ts": 904869.481921196,
      "dur": 706.9110870361328,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 1
      },
      "sourceIndex": 111
    },
    {
      "name": "backward-recv",
      "cat": "backward-recv",
      "pid": 0,
      "tid": 21,
      "ts": 1237567.0373439789,
      "dur": 715.4941558837891,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 2
      },
      "sourceIndex": 112
    },
    {
      "name": "backward-recv",
      "cat": "backward-recv",
      "pid": 0,
      "tid": 21,
      "ts": 1570920.5269813538,
      "dur": 374.79400634765625,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 3
      },
      "sourceIndex": 113
    },
    {
      "name": "backward-send",
      "cat": "backward-send",
      "pid": 0,
      "tid": 21,
      "ts": 710107.7114930376,
      "dur": 767.7078247070312,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 114
    },
    {
      "name": "backward-send",
      "cat": "backward-send",
      "pid": 0,
      "tid": 21,
      "ts": 1043502.1534794942,
      "dur": 776.2908935546875,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 1
      },
      "sourceIndex": 115
    },
    {
      "name": "backward-send",
      "cat": "backward-send",
      "pid": 0,
      "tid": 21,
      "ts": 1376048.9012347534,
      "dur": 386.23809814453125,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 2
      },
      "sourceIndex": 116
    },
    {
      "name": "backward-send",
      "cat": "backward-send",
      "pid": 0,
      "tid": 21,
      "ts": 1709107.1044269484,
      "dur": 391.96014404296875,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 3
      },
      "sourceIndex": 117
    },
    {
      "name": "forward-compute",
      "cat": "forward-compute",
      "pid": 0,
      "tid": 20,
      "ts": 176764.24980163574,
      "dur": 63044.92801427841,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 0,
        "seq_id": 0
      },
      "sourceIndex": 118
    },
    {
      "name": "forward-compute",
      "cat": "forward-compute",
      "pid": 0,
      "tid": 20,
      "ts": 242291.68891906738,
      "dur": 63503.58575582504,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 1,
        "seq_id": 1
      },
      "sourceIndex": 119
    },
    {
      "name": "forward-compute",
      "cat": "forward-compute",
      "pid": 0,
      "tid": 20,
      "ts": 710881.233215332,
      "dur": 65036.892890930176,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 2,
        "seq_id": 2
      },
      "sourceIndex": 120
    },
    {
      "name": "forward-compute",
      "cat": "forward-compute",
      "pid": 0,
      "tid": 20,
      "ts": 1044275.5222320557,
      "dur": 65379.008650779724,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 3,
        "seq_id": 3
      },
      "sourceIndex": 121
    },
    {
      "name": "forward-recv",
      "cat": "forward-recv",
      "pid": 0,
      "tid": 21,
      "ts": 176340.69919586182,
      "dur": 421.28562927246094,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 122
    },
    {
      "name": "forward-recv",
      "cat": "forward-recv",
      "pid": 0,
      "tid": 21,
      "ts": 241863.77180740237,
      "dur": 422.9545593261719,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 1
      },
      "sourceIndex": 123
    },
    {
      "name": "forward-recv",
      "cat": "forward-recv",
      "pid": 0,
      "tid": 21,
      "ts": 710107.7114930376,
      "dur": 767.7078247070312,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 2
      },
      "sourceIndex": 124
    },
    {
      "name": "forward-recv",
      "cat": "forward-recv",
      "pid": 0,
      "tid": 21,
      "ts": 1043502.1534794942,
      "dur": 776.2908935546875,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 3
      },
      "sourceIndex": 125
    },
    {
      "name": "forward-send",
      "cat": "forward-send",
      "pid": 0,
      "tid": 21,
      "ts": 239811.18606752716,
      "dur": 379.32395935058594,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 126
    },
    {
      "name": "forward-send",
      "cat": "forward-send",
      "pid": 0,
      "tid": 21,
      "ts": 571644.9022293091,
      "dur": 713.5868072509766,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 1
      },
      "sourceIndex": 127
    },
    {
      "name": "forward-send",
      "cat": "forward-send",
      "pid": 0,
      "tid": 21,
      "ts": 904869.481921196,
      "dur": 706.9110870361328,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 2
      },
      "sourceIndex": 128
    },
    {
      "name": "forward-send",
      "cat": "forward-send",
      "pid": 0,
      "tid": 21,
      "ts": 1237567.0373439789,
      "dur": 715.4941558837891,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 3
      },
      "sourceIndex": 129
    },
    {
      "name": "gc",
      "cat": "gc",
      "pid": 0,
      "tid": 20,
      "ts": 1709105.7300567627,
      "dur": 0,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": 0,
        "seq_id": 0
      },
      "sourceIndex": 130
    },
    {
      "name": "grads-reduce-scatter",
      "cat": "grads-reduce-scatter",
      "pid": 0,
      "tid": 23,
      "ts": 1726075.9137570858,
      "dur": 50266.50428771973,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 131
    },
    {
      "name": "layernorm-grads-all-reduce",
      "cat": "layernorm-grads-all-reduce",
      "pid": 0,
      "tid": 23,
      "ts": 1710833.7879180908,
      "dur": 64.73599933087826,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 132
    },
    {
      "name": "optimizer",
      "cat": "optimizer",
      "pid": 0,
      "tid": 20,
      "ts": 2338286.3998413086,
      "dur": 37979.841232299805,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 133
    },
    {
      "name": "optimizer-clip-main-grad",
      "cat": "optimizer-clip-main-grad",
      "pid": 0,
      "tid": 23,
      "ts": 2324741.9595718384,
      "dur": 13544.559478759766,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 134
    },
    {
      "name": "params-all-gather",
      "cat": "params-all-gather",
      "pid": 0,
      "tid": 23,
      "ts": 190.33066928386688,
      "dur": 24417.400360107422,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 135
    },
    {
      "name": "backward-compute",
      "cat": "backward-compute",
      "pid": 1,
      "tid": 25,
      "ts": 563329.6966552734,
      "dur": 133375.07843971252,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 0,
        "seq_id": 0
      },
      "sourceIndex": 136
    },
    {
      "name": "backward-compute",
      "cat": "backward-compute",
      "pid": 1,
      "tid": 25,
      "ts": 897359.3711853027,
      "dur": 137518.65923404694,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 1,
        "seq_id": 1
      },
      "sourceIndex": 137
    },
    {
      "name": "backward-compute",
      "cat": "backward-compute",
      "pid": 1,
      "tid": 25,
      "ts": 1226119.7566986084,
      "dur": 133026.01873874664,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 2,
        "seq_id": 2
      },
      "sourceIndex": 138
    },
    {
      "name": "backward-compute",
      "cat": "backward-compute",
      "pid": 1,
      "tid": 25,
      "ts": 1556591.0339355469,
      "dur": 133164.70384597778,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 3,
        "seq_id": 3
      },
      "sourceIndex": 139
    },
    {
      "name": "backward-recv",
      "cat": "backward-recv",
      "pid": 1,
      "tid": 26,
      "ts": 562593.3408737183,
      "dur": 745.0580596923828,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 140
    },
    {
      "name": "backward-recv",
      "cat": "backward-recv",
      "pid": 1,
      "tid": 26,
      "ts": 896628.5288333893,
      "dur": 726.9382476806641,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 1
      },
      "sourceIndex": 141
    },
    {
      "name": "backward-recv",
      "cat": "backward-recv",
      "pid": 1,
      "tid": 26,
      "ts": 1225382.4770450592,
      "dur": 735.7597351074219,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 2
      },
      "sourceIndex": 142
    },
    {
      "name": "backward-recv",
      "cat": "backward-recv",
      "pid": 1,
      "tid": 26,
      "ts": 1556198.9843845367,
      "dur": 388.3838653564453,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 3
      },
      "sourceIndex": 143
    },
    {
      "name": "backward-send",
      "cat": "backward-send",
      "pid": 1,
      "tid": 26,
      "ts": 696707.9592868686,
      "dur": 778.4366607666016,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 144
    },
    {
      "name": "backward-send",
      "cat": "backward-send",
      "pid": 1,
      "tid": 26,
      "ts": 1034882.215608377,
      "dur": 762.939453125,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 1
      },
      "sourceIndex": 145
    },
    {
      "name": "backward-send",
      "cat": "backward-send",
      "pid": 1,
      "tid": 26,
      "ts": 1359154.3890943285,
      "dur": 391.7217254638672,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 2
      },
      "sourceIndex": 146
    },
    {
      "name": "backward-send",
      "cat": "backward-send",
      "pid": 1,
      "tid": 26,
      "ts": 1689758.317108499,
      "dur": 385.9996795654297,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 3
      },
      "sourceIndex": 147
    },
    {
      "name": "forward-compute",
      "cat": "forward-compute",
      "pid": 1,
      "tid": 25,
      "ts": 174003.12423706055,
      "dur": 61336.61046624184,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 0,
        "seq_id": 0
      },
      "sourceIndex": 148
    },
    {
      "name": "forward-compute",
      "cat": "forward-compute",
      "pid": 1,
      "tid": 25,
      "ts": 238674.64065551758,
      "dur": 63787.490129470825,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 1,
        "seq_id": 1
      },
      "sourceIndex": 149
    },
    {
      "name": "forward-compute",
      "cat": "forward-compute",
      "pid": 1,
      "tid": 25,
      "ts": 697481.3938140869,
      "dur": 62774.911522865295,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 2,
        "seq_id": 2
      },
      "sourceIndex": 150
    },
    {
      "name": "forward-compute",
      "cat": "forward-compute",
      "pid": 1,
      "tid": 25,
      "ts": 1035647.8691101074,
      "dur": 63981.3095331192,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 3,
        "seq_id": 3
      },
      "sourceIndex": 151
    },
    {
      "name": "forward-recv",
      "cat": "forward-recv",
      "pid": 1,
      "tid": 26,
      "ts": 173558.77161026,
      "dur": 439.1670227050781,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 152
    },
    {
      "name": "forward-recv",
      "cat": "forward-recv",
      "pid": 1,
      "tid": 26,
      "ts": 238226.3196632266,
      "dur": 442.5048828125,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 1
      },
      "sourceIndex": 153
    },
    {
      "name": "forward-recv",
      "cat": "forward-recv",
      "pid": 1,
      "tid": 26,
      "ts": 696707.9592868686,
      "dur": 778.4366607666016,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 2
      },
      "sourceIndex": 154
    },
    {
      "name": "forward-recv",
      "cat": "forward-recv",
      "pid": 1,
      "tid": 26,
      "ts": 1034882.215608377,
      "dur": 762.939453125,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 3
      },
      "sourceIndex": 155
    },
    {
      "name": "forward-send",
      "cat": "forward-send",
      "pid": 1,
      "tid": 26,
      "ts": 235342.83650224097,
      "dur": 376.2245178222656,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 156
    },
    {
      "name": "forward-send",
      "cat": "forward-send",
      "pid": 1,
      "tid": 26,
      "ts": 562593.3408737183,
      "dur": 745.0580596923828,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 1
      },
      "sourceIndex": 157
    },
    {
      "name": "forward-send",
      "cat": "forward-send",
      "pid": 1,
      "tid": 26,
      "ts": 896628.5288333893,
      "dur": 726.9382476806641,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 2
      },
      "sourceIndex": 158
    },
    {
      "name": "forward-send",
      "cat": "forward-send",
      "pid": 1,
      "tid": 26,
      "ts": 1225382.4770450592,
      "dur": 735.7597351074219,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 3
      },
      "sourceIndex": 159
    },
    {
      "name": "gc",
      "cat": "gc",
      "pid": 1,
      "tid": 25,
      "ts": 1689755.6781768799,
      "dur": 0,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": 0,
        "seq_id": 0
      },
      "sourceIndex": 160
    },
    {
      "name": "grads-reduce-scatter",
      "cat": "grads-reduce-scatter",
      "pid": 1,
      "tid": 28,
      "ts": 1726075.8206248283,
      "dur": 50290.584564208984,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 161
    },
    {
      "name": "layernorm-grads-all-reduce",
      "cat": "layernorm-grads-all-reduce",
      "pid": 1,
      "tid": 28,
      "ts": 1691266.2982940674,
      "dur": 38.144000427564606,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 162
    },
    {
      "name": "optimizer",
      "cat": "optimizer",
      "pid": 1,
      "tid": 25,
      "ts": 2338262.0811462402,
      "dur": 38079.261779785156,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 163
    },
    {
      "name": "optimizer-clip-main-grad",
      "cat": "optimizer-clip-main-grad",
      "pid": 1,
      "tid": 28,
      "ts": 2324741.9595718384,
      "dur": 13520.240783691406,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 164
    },
    {
      "name": "params-all-gather",
      "cat": "params-all-gather",
      "pid": 1,
      "tid": 28,
      "ts": 190.32135605812073,
      "dur": 24430.99021911621,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 165
    },
    {
      "name": "backward-compute",
      "cat": "backward-compute",
      "pid": 0,
      "tid": 30,
      "ts": 374353.17039489746,
      "dur": 197291.92554950714,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 0,
        "seq_id": 0
      },
      "sourceIndex": 166
    },
    {
      "name": "backward-compute",
      "cat": "backward-compute",
      "pid": 0,
      "tid": 30,
      "ts": 709202.766418457,
      "dur": 195664.31641578674,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 1,
        "seq_id": 1
      },
      "sourceIndex": 167
    },
    {
      "name": "backward-compute",
      "cat": "backward-compute",
      "pid": 0,
      "tid": 30,
      "ts": 1042243.7191009521,
      "dur": 195321.187376976,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 2,
        "seq_id": 2
      },
      "sourceIndex": 168
    },
    {
      "name": "backward-compute",
      "cat": "backward-compute",
      "pid": 0,
      "tid": 30,
      "ts": 1375191.2117004395,
      "dur": 195727.25892066956,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 3,
        "seq_id": 3
      },
      "sourceIndex": 169
    },
    {
      "name": "backward-send",
      "cat": "backward-send",
      "pid": 0,
      "tid": 31,
      "ts": 571645.1004846022,
      "dur": 771.7609405517578,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 170
    },
    {
      "name": "backward-send",
      "cat": "backward-send",
      "pid": 0,
      "tid": 31,
      "ts": 904869.4649827667,
      "dur": 762.939453125,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 1
      },
      "sourceIndex": 171
    },
    {
      "name": "backward-send",
      "cat": "backward-send",
      "pid": 0,
      "tid": 31,
      "ts": 1237566.8923486955,
      "dur": 775.5756378173828,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 2
      },
      "sourceIndex": 172
    },
    {
      "name": "backward-send",
      "cat": "backward-send",
      "pid": 0,
      "tid": 31,
      "ts": 1570920.3673177399,
      "dur": 380.51605224609375,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 3
      },
      "sourceIndex": 173
    },
    {
      "name": "embedding-grads-all-reduce",
      "cat": "embedding-grads-all-reduce",
      "pid": 0,
      "tid": 33,
      "ts": 1986141.2048339844,
      "dur": 266367.19703674316,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 174
    },
    {
      "name": "forward-compute",
      "cat": "forward-compute",
      "pid": 0,
      "tid": 30,
      "ts": 240346.19331359863,
      "dur": 134002.2087097168,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 0,
        "seq_id": 0
      },
      "sourceIndex": 175
    },
    {
      "name": "forward-compute",
      "cat": "forward-compute",
      "pid": 0,
      "tid": 30,
      "ts": 572414.6366119385,
      "dur": 136783.8978767395,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 1,
        "seq_id": 1
      },
      "sourceIndex": 176
    },
    {
      "name": "forward-compute",
      "cat": "forward-compute",
      "pid": 0,
      "tid": 30,
      "ts": 905635.3569030762,
      "dur": 136602.58054733276,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 2,
        "seq_id": 2
      },
      "sourceIndex": 177
    },
    {
      "name": "forward-compute",
      "cat": "forward-compute",
      "pid": 0,
      "tid": 30,
      "ts": 1238336.3246917725,
      "dur": 136851.2213230133,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 3,
        "seq_id": 3
      },
      "sourceIndex": 178
    },
    {
      "name": "forward-recv",
      "cat": "forward-recv",
      "pid": 0,
      "tid": 31,
      "ts": 239811.28633022308,
      "dur": 530.7197570800781,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 179
    },
    {
      "name": "forward-recv",
      "cat": "forward-recv",
      "pid": 0,
      "tid": 31,
      "ts": 571645.1004846022,
      "dur": 771.7609405517578,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 1
      },
      "sourceIndex": 180
    },
    {
      "name": "forward-recv",
      "cat": "forward-recv",
      "pid": 0,
      "tid": 31,
      "ts": 904869.4649827667,
      "dur": 762.939453125,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 2
      },
      "sourceIndex": 181
    },
    {
      "name": "forward-recv",
      "cat": "forward-recv",
      "pid": 0,
      "tid": 31,
      "ts": 1237566.8923486955,
      "dur": 775.5756378173828,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 3
      },
      "sourceIndex": 182
    },
    {
      "name": "gc",
      "cat": "gc",
      "pid": 0,
      "tid": 30,
      "ts": 1570918.5600280762,
      "dur": 0,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": 0,
        "seq_id": 0
      },
      "sourceIndex": 183
    },
    {
      "name": "grads-reduce-scatter",
      "cat": "grads-reduce-scatter",
      "pid": 0,
      "tid": 33,
      "ts": 2269271.8394100666,
      "dur": 54425.00114440918,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 184
    },
    {
      "name": "layernorm-grads-all-reduce",
      "cat": "layernorm-grads-all-reduce",
      "pid": 0,
      "tid": 33,
      "ts": 1578307.1517944336,
      "dur": 30.68799924221821,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 185
    },
    {
      "name": "optimizer",
      "cat": "optimizer",
      "pid": 0,
      "tid": 30,
      "ts": 2339251.2798309326,
      "dur": 42144.775390625,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 186
    },
    {
      "name": "optimizer-clip-main-grad",
      "cat": "optimizer-clip-main-grad",
      "pid": 0,
      "tid": 33,
      "ts": 2324741.9502586126,
      "dur": 14509.439468383789,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 187
    },
    {
      "name": "params-all-gather",
      "cat": "params-all-gather",
      "pid": 0,
      "tid": 33,
      "ts": 5311.792716383934,
      "dur": 24290.800094604492,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 188
    },
    {
      "name": "backward-compute",
      "cat": "backward-compute",
      "pid": 1,
      "tid": 35,
      "ts": 368464.23149108887,
      "dur": 194121.21176719666,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 0,
        "seq_id": 0
      },
      "sourceIndex": 189
    },
    {
      "name": "backward-compute",
      "cat": "backward-compute",
      "pid": 1,
      "tid": 35,
      "ts": 700253.4866333008,
      "dur": 196371.55532836914,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 1,
        "seq_id": 1
      },
      "sourceIndex": 190
    },
    {
      "name": "backward-compute",
      "cat": "backward-compute",
      "pid": 1,
      "tid": 35,
      "ts": 1032488.1076812744,
      "dur": 192887.76814937592,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 2,
        "seq_id": 2
      },
      "sourceIndex": 191
    },
    {
      "name": "backward-compute",
      "cat": "backward-compute",
      "pid": 1,
      "tid": 35,
      "ts": 1361601.3526916504,
      "dur": 194595.605134964,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 3,
        "seq_id": 3
      },
      "sourceIndex": 192
    },
    {
      "name": "backward-send",
      "cat": "backward-send",
      "pid": 1,
      "tid": 36,
      "ts": 562593.1639224291,
      "dur": 782.4897766113281,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 193
    },
    {
      "name": "backward-send",
      "cat": "backward-send",
      "pid": 1,
      "tid": 36,
      "ts": 896628.5344795324,
      "dur": 766.5157318115234,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 1
      },
      "sourceIndex": 194
    },
    {
      "name": "backward-send",
      "cat": "backward-send",
      "pid": 1,
      "tid": 36,
      "ts": 1225382.645148784,
      "dur": 776.5293121337891,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 2
      },
      "sourceIndex": 195
    },
    {
      "name": "backward-send",
      "cat": "backward-send",
      "pid": 1,
      "tid": 36,
      "ts": 1556199.131620815,
      "dur": 387.43019104003906,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 3
      },
      "sourceIndex": 196
    },
    {
      "name": "embedding-grads-all-reduce",
      "cat": "embedding-grads-all-reduce",
      "pid": 1,
      "tid": 38,
      "ts": 1960034.0723991394,
      "dur": 289998.53134155273,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 197
    },
    {
      "name": "forward-compute",
      "cat": "forward-compute",
      "pid": 1,
      "tid": 35,
      "ts": 235858.67881774902,
      "dur": 132603.54101657867,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 0,
        "seq_id": 0
      },
      "sourceIndex": 198
    },
    {
      "name": "forward-compute",
      "cat": "forward-compute",
      "pid": 1,
      "tid": 35,
      "ts": 563370.7046508789,
      "dur": 136879.3398141861,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 1,
        "seq_id": 1
      },
      "sourceIndex": 199
    },
    {
      "name": "forward-compute",
      "cat": "forward-compute",
      "pid": 1,
      "tid": 35,
      "ts": 897390.3656005859,
      "dur": 135094.09129619598,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 2,
        "seq_id": 2
      },
      "sourceIndex": 200
    },
    {
      "name": "forward-compute",
      "cat": "forward-compute",
      "pid": 1,
      "tid": 35,
      "ts": 1226160.0494384766,
      "dur": 135434.88085269928,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": 3,
        "seq_id": 3
      },
      "sourceIndex": 201
    },
    {
      "name": "forward-recv",
      "cat": "forward-recv",
      "pid": 1,
      "tid": 36,
      "ts": 235342.8155183792,
      "dur": 512.1231079101562,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 202
    },
    {
      "name": "forward-recv",
      "cat": "forward-recv",
      "pid": 1,
      "tid": 36,
      "ts": 562593.1639224291,
      "dur": 782.4897766113281,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 1
      },
      "sourceIndex": 203
    },
    {
      "name": "forward-recv",
      "cat": "forward-recv",
      "pid": 1,
      "tid": 36,
      "ts": 896628.5344795324,
      "dur": 766.5157318115234,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 2
      },
      "sourceIndex": 204
    },
    {
      "name": "forward-recv",
      "cat": "forward-recv",
      "pid": 1,
      "tid": 36,
      "ts": 1225382.645148784,
      "dur": 776.5293121337891,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 3
      },
      "sourceIndex": 205
    },
    {
      "name": "gc",
      "cat": "gc",
      "pid": 1,
      "tid": 35,
      "ts": 1556196.928024292,
      "dur": 0,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": 0,
        "seq_id": 0
      },
      "sourceIndex": 206
    },
    {
      "name": "grads-reduce-scatter",
      "cat": "grads-reduce-scatter",
      "pid": 1,
      "tid": 38,
      "ts": 2269271.858036518,
      "dur": 54412.6033782959,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 207
    },
    {
      "name": "layernorm-grads-all-reduce",
      "cat": "layernorm-grads-all-reduce",
      "pid": 1,
      "tid": 38,
      "ts": 1575726.5090942383,
      "dur": 31.807998311705887,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 208
    },
    {
      "name": "optimizer",
      "cat": "optimizer",
      "pid": 1,
      "tid": 35,
      "ts": 2339252.9487609863,
      "dur": 42399.16801452637,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 209
    },
    {
      "name": "optimizer-clip-main-grad",
      "cat": "optimizer-clip-main-grad",
      "pid": 1,
      "tid": 38,
      "ts": 2324741.7444363236,
      "dur": 14511.1083984375,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": -1,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 210
    },
    {
      "name": "params-all-gather",
      "cat": "params-all-gather",
      "pid": 1,
      "tid": 38,
      "ts": 5311.833694577217,
      "dur": 24304.628372192383,
      "ph": "X",
      "args": {
        "step": 23,
        "model_chunk": 0,
        "mb_id": -1,
        "seq_id": 0
      },
      "sourceIndex": 211
    }
  ]
};
if (typeof module !== "undefined") module.exports = data; else root.STStep23 = data;
})(globalThis);

