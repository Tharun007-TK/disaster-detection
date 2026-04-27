// Legacy compatibility shim — the top-bar Nav has been replaced by Sidebar.
// Any lingering imports of `@/components/Nav` resolve to the new sidebar.
export { default } from "./Sidebar";
