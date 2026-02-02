import { subMinutes } from "date-fns";

export type RiskLevel = "HIGH" | "MEDIUM" | "LOW";
export type Language = "en" | "es" | "fr" | "ar" | "zh" | "ht"; // Added ht for Haitian Creole

export interface Classification {
  category: string;
  confidence: number; // 0 to 1
}

export interface SignalItem {
  id: string;
  timestamp: Date;
  source: string; // "Direct Report", "News", "Twitter", "Telegram", etc.
  content: string;
  originalContent?: string; // If translated
  language: Language;
  riskLevel: RiskLevel;
  categories: string[]; // Top 3
  classifications: Classification[]; // All detected
  isTranslated: boolean;
}

export interface ModelInfo {
  version: string;
  f1_score: number | null;
  status: string;
  hierarchy_violations: number;
}

/** Category group name -> list of display category names. Runtime source: GET /api/categories. */
export type CategoryGroups = Record<string, string[]>;

/** Fallback for initial render and mock data only; runtime source is GET /api/categories. */
export const DEFAULT_CATEGORY_GROUPS: CategoryGroups = {
  "Critical Needs": [
    "Medical Help", "Medical Products", "Search & Rescue", "Water", "Food",
    "Shelter", "Security", "Hospitals", "Missing People", "Refugees", "Death"
  ],
  "Infrastructure": [
    "Transport", "Buildings", "Electricity", "Tools", "Shops",
    "Aid Centers", "Other Infrastructure"
  ],
  "Weather": [
    "Floods", "Storm", "Fire", "Earthquake", "Cold", "Other Weather"
  ],
  "Other": [
    "Clothing", "Money",
    "Other Aid", "Military", "Child Alone", "Request", "Offer", "Direct Report"
  ]
};

export const ALL_CATEGORIES = Object.values(DEFAULT_CATEGORY_GROUPS).flat();

export const CRITICAL_CATEGORIES = DEFAULT_CATEGORY_GROUPS["Critical Needs"];

// Helper to calculate severity
const calculateSeverity = (classifications: Classification[]): RiskLevel => {
  const criticalCount = classifications.filter(c => 
    CRITICAL_CATEGORIES.includes(c.category) && c.confidence > 0.5
  ).length;
  
  const maxConf = Math.max(...classifications.map(c => c.confidence));

  if (criticalCount >= 2 || maxConf > 0.85) return "HIGH";
  if (criticalCount >= 1 || maxConf > 0.70) return "MEDIUM";
  return "LOW";
};

// Mock Messages
const MOCK_MESSAGES = [
  { text: "Urgent: Water rising rapidly in downtown area. People trapped on roofs near the market. Need rescue immediately!", cats: ["Floods", "Search & Rescue", "Water"] },
  { text: "Hospital generator failed. We have 3 critical patients on ventilators. Need fuel or backup power ASAP.", cats: ["Hospitals", "Electricity", "Medical Help"] },
  { text: "Road to north village blocked by landslide. No access for supply trucks.", cats: ["Transport", "Other Infrastructure"] },
  { text: "Family of 4 missing since the tremor. Last seen near the old church.", cats: ["Missing People", "Earthquake"] },
  { text: "Distribution center established at the school. We have water and blankets but need baby formula.", cats: ["Aid Centers", "Food", "Medical Products"] },
  { text: "Looting reported in sector 4. Shops are being broken into. Security needed.", cats: ["Security", "Shops", "Civil Unrest"] },
  { text: "Severe cold front approaching tonight. Need warm clothing and heaters for the shelter.", cats: ["Cold", "Shelter", "Clothing"] },
  { text: "Bridge collapsed on Main St. Cars fell into the river.", cats: ["Transport", "Infrastructure Failure", "Search & Rescue"] },
  { text: "Anyone have insulin? My supply was lost in the flood.", cats: ["Medical Products", "Floods", "Request"] },
  { text: "Military trucks seen moving towards the coast. Relief effort starting?", cats: ["Military", "Direct Report"] },
  { text: "Is there any food distribution today? We haven't eaten in 2 days.", cats: ["Food", "Request"] },
  { text: "Fire spreading near the fuel depot. Evacuate immediately!", cats: ["Fire", "Security"] },
  { text: "Found a lost child, approx 5 years old, wearing red shirt. At the community center.", cats: ["Child Alone", "Refugees"] },
  { text: "Need help moving debris from my house. trapped inside.", cats: ["Search & Rescue", "Buildings"] },
  { text: "Water supply contaminated. Do not drink from the tap.", cats: ["Water", "Health Emergency"] },
  // Creole examples (translated)
  { text: "We need clean water and tents. The rain won't stop.", cats: ["Water", "Shelter", "Storm"], lang: "ht", original: "Nou bezwen dlo pwòp ak tant. Lapli a pap kanpe." },
  { text: "My leg is broken. Cannot walk to the clinic.", cats: ["Medical Help", "Transport"], lang: "ht", original: "Janm mwen kase. Mwen pa ka mache ale nan klinik la." }
];

const generateSignals = (count: number): SignalItem[] => {
  return Array.from({ length: count }).map((_, i) => {
    const template = MOCK_MESSAGES[Math.floor(Math.random() * MOCK_MESSAGES.length)];
    const isTranslated = template.lang === 'ht';
    const timestamp = subMinutes(new Date(), Math.floor(Math.random() * 360)); // Last 6 hours
    
    // Simulate classifications
    const classifications = template.cats.map(cat => ({
      category: cat,
      confidence: 0.6 + (Math.random() * 0.35) // 0.6 to 0.95
    }));

    // Add some noise
    if (Math.random() > 0.7) {
      classifications.push({
        category: ALL_CATEGORIES[Math.floor(Math.random() * ALL_CATEGORIES.length)],
        confidence: 0.3 + (Math.random() * 0.3)
      });
    }

    const severity = calculateSeverity(classifications);
    
    // Sort classifications by confidence
    classifications.sort((a, b) => b.confidence - a.confidence);

    return {
      id: `SIG-${1000 + i}`,
      timestamp,
      source: ["Twitter", "Facebook", "Telegram", "SMS", "Direct Report", "News"][Math.floor(Math.random() * 6)],
      content: template.text,
      originalContent: template.original,
      language: (template.lang || "en") as Language,
      riskLevel: severity,
      categories: classifications.slice(0, 3).map(c => c.category),
      classifications,
      isTranslated
    };
  }).sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
};

export const MOCK_SIGNALS = generateSignals(45);

export const SYSTEM_METRICS = {
  volumeToday: 2450,
  flaggedRate: 12.5,
  flaggedHistory: [
    { time: "6h ago", count: 45 },
    { time: "5h ago", count: 120 },
    { time: "4h ago", count: 80 },
    { time: "3h ago", count: 210 },
    { time: "2h ago", count: 150 },
    { time: "1h ago", count: 95 },
    { time: "Now", count: 60 }
  ],
  topCategories: [
    { name: "Medical Help", count: 342 },
    { name: "Water", count: 285 },
    { name: "Shelter", count: 190 },
    { name: "Food", count: 154 },
    { name: "Search & Rescue", count: 120 },
    { name: "Floods", count: 98 }
  ]
};

export const getCategoryVolume = (categoryName: string): number => {
  const top = SYSTEM_METRICS.topCategories.find(c => c.name === categoryName);
  if (top) return top.count;
  // Deterministic mock for others based on name length
  return 50 + (categoryName.length * 12) % 150;
};
