import axios from "axios";

export const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
export const API_KEY = import.meta.env.VITE_API_KEY || "";
export const WS_URL = API_URL.replace(/^http/, "ws") + "/video-feed";

// Single axios instance so the API key header lives in one place instead
// of being duplicated across every page component.
export const api = axios.create({
  baseURL: API_URL,
  headers: API_KEY ? { "X-API-Key": API_KEY } : {},
});

// The WebSocket handshake can't carry custom headers, so the key travels
// as a query param instead — only appended if one is configured.
export function wsUrlWithAuth() {
  return API_KEY ? `${WS_URL}?api_key=${encodeURIComponent(API_KEY)}` : WS_URL;
}
