import { Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import LiveAnalysis from "./pages/LiveAnalysis";
import Analytics from "./pages/Analytics";
import Navbar from "./components/Navbar";

export default function App() {
  return (
    <div className="min-h-screen" style={{ backgroundColor: "#F0F9FF" }}>
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/live" element={<LiveAnalysis />} />
        <Route path="/analytics" element={<Analytics />} />
      </Routes>
    </div>
  );
}
