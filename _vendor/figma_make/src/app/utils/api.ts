/**
 * Maps display names (as shown in the UI) to API internal names (as expected by the server).
 * 
 * This ensures that when users interact with category filters using display names,
 * the correct internal category names are sent to the API endpoints.
 */
export function toApiName(displayName: string): string {
  // Special mappings for categories with non-standard display names
  if (displayName === "Search & Rescue") return "search_and_rescue";
  if (displayName === "Infrastructure") return "infrastructure_related";
  if (displayName === "Other Infrastructure") return "other_infrastructure";
  if (displayName === "Other Weather") return "other_weather";
  if (displayName === "Other Aid") return "other_aid";
  if (displayName === "Medical Help") return "medical_help";
  if (displayName === "Medical Products") return "medical_products";
  if (displayName === "Aid Centers") return "aid_centers";
  if (displayName === "Child Alone") return "child_alone";
  if (displayName === "Direct Report") return "direct_report";
  if (displayName === "Missing People") return "missing_people";
  
  // Default: convert to lowercase and replace spaces with underscores
  return displayName.toLowerCase().replace(/ /g, "_");
}
