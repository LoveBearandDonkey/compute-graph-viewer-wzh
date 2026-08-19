window.ReportRuntimeConfig = {
  "templateVersion": 2,
  "operatorDetails": "./outputs/operator_details.json",
  "hbm": "./outputs/hbm_series.json",
  "findings": "./outputs/metrics_findings.json",
  "expertInventory": "./outputs/expert_inventory.json",
  "analysis": "../ds3.2-exp-w8a8c8_analysis_config.json",
  "performance": "../ds3.2-exp-w8a8c8_perf_data.json",
  "timeline": "../ds3.2-exp-w8a8c8_timeline.json",
  "trace": "../trace_view.json",
  "bindings": "./outputs/trace_bindings.json",
  "architecture": "./outputs/model_architecture_graph.json",
  "overlay": "./outputs/architecture_overlay_map.json",
  "provenance": {
    "skills": [
      "1-perf-breakdown",
      "2-adapt-breakdown-to-ui-json"
    ],
    "modelSource": "models_src/modeling_deepseek.py; models_src/configuration_deepseek.py",
    "extractorModel": "Claude",
    "status": "mock"
  },
  "capabilities": {},
  "templateOverrides": []
};
