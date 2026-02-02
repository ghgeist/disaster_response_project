/** Category groups returned by GET /api/categories (single source of truth). */
export interface CategoriesResponse {
  categories: { internal: string; display: string; count: number }[];
  groups: Record<string, string[]>;
}

/** Fetches category metadata and groups from the backend (single source of truth for CATEGORY_GROUPS). */
export async function getCategories(): Promise<CategoriesResponse> {
  const res = await fetch("/api/categories");
  if (!res.ok) throw new Error(`Categories ${res.status}`);
  return res.json();
}

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
  
  // Default: convert to lowercase and replace spaces with underscores
  return displayName.toLowerCase().replace(/ /g, "_");
}
