import { NavLink } from "react-router-dom";

export default function Navbar() {
  const linkClass = ({ isActive }) =>
    isActive
      ? "px-4 py-1.5 rounded-full text-sm font-medium transition-all duration-200"
      : "px-4 py-1.5 rounded-full text-sm font-medium transition-all duration-200";

  return (
    <nav
      style={{
        backgroundColor: "#c7e8fd",
        borderBottom: "1px solid #BAE6FD",
      }}
      className="sticky top-0 z-50"
    >
      <div className="max-w-screen-xl mx-auto px-6 py-3 flex items-center justify-between">
        {/* Left: Logo + Title */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span
              className="w-2.5 h-2.5 rounded-full inline-block"
              style={{ backgroundColor: "#0284C7" }}
            />
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="#0284C7"
              strokeWidth="1.8"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M15 10l4.553-2.069A1 1 0 0121 8.82V15.18a1 1 0 01-1.447.89L15 14M3 8a2 2 0 012-2h10a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z" />
            </svg>
          </div>
          <span
            className="font-bold text-base tracking-tight"
            style={{ color: "#0C4A6E" }}
          >
            Indian Road Intelligence System
          </span>
        </div>

        {/* Right: Nav Links */}
        <div className="flex items-center gap-1">
          <NavLink
            to="/"
            end
            className={({ isActive }) =>
              isActive
                ? "px-4 py-1.5 rounded-full text-sm font-medium transition-all duration-200 text-sky-lightest"
                : "px-4 py-1.5 rounded-full text-sm font-medium transition-all duration-200"
            }
            style={({ isActive }) =>
              isActive
                ? { backgroundColor: "#0284C7", color: "#F0F9FF" }
                : { color: "#0284C7" }
            }
          >
            Upload
          </NavLink>
          <NavLink
            to="/live"
            className={({ isActive }) =>
              isActive
                ? "px-4 py-1.5 rounded-full text-sm font-medium transition-all duration-200"
                : "px-4 py-1.5 rounded-full text-sm font-medium transition-all duration-200"
            }
            style={({ isActive }) =>
              isActive
                ? { backgroundColor: "#0284C7", color: "#F0F9FF" }
                : { color: "#0284C7" }
            }
          >
            Live Analysis
          </NavLink>
          <NavLink
            to="/analytics"
            className={({ isActive }) =>
              isActive
                ? "px-4 py-1.5 rounded-full text-sm font-medium transition-all duration-200"
                : "px-4 py-1.5 rounded-full text-sm font-medium transition-all duration-200"
            }
            style={({ isActive }) =>
              isActive
                ? { backgroundColor: "#0284C7", color: "#F0F9FF" }
                : { color: "#0284C7" }
            }
          >
            Analytics
          </NavLink>
        </div>
      </div>
    </nav>
  );
}
