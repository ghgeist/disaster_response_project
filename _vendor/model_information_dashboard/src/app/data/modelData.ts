
export const MODEL_METRICS = {
  f1Score: 93.8,
  precision: 94.2,
  recall: 93.5,
  id: "disaster_rf_prod_2026-01-22",
  version: "2.4.0",
  lastUpdated: "2026-02-02T14:30:00Z",
  description: "Balanced Class Weighting fixed the 0% recall failure for critical categories."
};

export const REGISTRY_FILES = [
  { name: "MODEL_INFO.json", size: "2.4 KB", type: "json" },
  { name: "performance_metrics.csv", size: "48 KB", type: "csv" },
  { name: "optimized_critical_thresholds.json", size: "1.2 KB", highlight: true, type: "json" },
  { name: "disaster_rf_prod_2026-01-22_thresholds.json", size: "3.1 KB", type: "json" },
  { name: "README.md", size: "4.5 KB", type: "md" }
];

// Map category names to their specific critical thresholds (0.0 - 1.0)
export const THRESHOLD_MAP: Record<string, number> = {
  "Medical Help": 0.82,
  "Search & Rescue": 0.85,
  "Water Rescue": 0.88,
  "Fire Evac": 0.84,
  "Collapse": 0.86,
  "Structural Defect": 0.85,
  "Cyber Attack": 0.90,
  "Nuclear": 0.95
};

export const CRITICAL_THRESHOLDS = Object.entries(THRESHOLD_MAP).map(([label, value]) => ({ label, value }));

export const HIERARCHY_SPECS = [
  {
    parent: "Critical Response",
    children: ["Medical Help", "Search & Rescue", "Water Rescue", "Fire Evac"]
  },
  {
    parent: "Geophysical",
    children: ["Earthquake", "Volcano", "Landslide", "Tsunami"]
  },
  {
    parent: "Meteorological",
    children: ["Storm", "Hurricane", "Tornado", "Snowstorm", "Hail"]
  },
  {
    parent: "Infrastructure",
    children: ["Collapse", "Structural Defect", "Transport", "Power Outage"]
  },
  {
    parent: "Security",
    children: ["Explosion", "Terrorism", "Cyber Attack", "War"]
  }
];

export const CATEGORIES = [
  { name: "Shops", f1: 0.01, count: 26 },
  { name: "Offer", f1: 0.02, count: 29 },
  { name: "Missing People", f1: 0.02, count: 44 },
  { name: "Tools", f1: 0.02, count: 28 },
  { name: "Security", f1: 0.05, count: 95 },
  { name: "Hospitals", f1: 0.07, count: 53 },
  { name: "Aid Centers", f1: 0.08, count: 58 },
  { name: "Search & Rescue", f1: 0.09, count: 138 },
  { name: "Fire", f1: 0.19, count: 62 },
  { name: "Transport", f1: 0.22, count: 245 },
  { name: "Other Infrastructure", f1: 0.25, count: 247 },
  { name: "Infrastructure Related", f1: 0.30, count: 352 },
  { name: "Other Weather", f1: 0.34, count: 285 },
  { name: "Refugees", f1: 0.42, count: 162 },
  { name: "Other Aid", f1: 0.42, count: 683 },
  { name: "Money", f1: 0.46, count: 110 },
  { name: "Electricity", f1: 0.47, count: 119 },
  { name: "Medical Products", f1: 0.48, count: 273 },
  { name: "Medical Help", f1: 0.54, count: 432 },
  { name: "Buildings", f1: 0.54, count: 289 },
  { name: "Cold", f1: 0.58, count: 105 },
  { name: "Clothing", f1: 0.60, count: 89 },
  { name: "Direct Report", f1: 0.62, count: 1002 },
  { name: "Death", f1: 0.64, count: 242 },
  { name: "Floods", f1: 0.65, count: 437 }
];

export const SYSTEM_BOUNDARIES = {
  does: [
    "Multi-label triage",
    "Severity estimation",
    "Resource allocation logic"
  ],
  doesNot: [
    "Ground truth verification",
    "Geolocation (High precision)",
    "Personal Identification (PII)"
  ]
};
