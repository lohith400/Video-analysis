/**
 * Utility to parse Indian Road Intelligence System traffic_log.csv files.
 * Handles both the time-series snapshot rows and the structured
 * '===== FINAL SESSION REPORT =====' multi-section report.
 */

export function parseTrafficLogCSV(csvText) {
  if (!csvText || typeof csvText !== "string") {
    return { timeSeries: [], report: null };
  }

  const lines = csvText.split(/\r?\n/);
  const timeSeries = [];
  let report = null;

  let inFinalReport = false;
  let currentSection = null;
  let reportData = {
    generatedAt: null,
    vehicleCounts: {},
    humans: { total: 0, males: 0, females: 0, children: 0, unknown: 0 },
    plates: [],
    twoWheelers: [],
    perVehicle: []
  };

  let headers = null;

  for (let i = 0; i < lines.length; i++) {
    const rawLine = lines[i].trim();
    if (!rawLine) continue;

    if (rawLine.includes("===== FINAL SESSION REPORT =====")) {
      inFinalReport = true;
      currentSection = null;
      continue;
    }

    if (!inFinalReport) {
      // Parsing Time Series Data
      const parts = parseCSVLine(rawLine);
      if (!headers) {
        headers = parts.map(h => h.trim().toLowerCase());
        continue;
      }

      if (parts.length >= headers.length) {
        const row = {};
        headers.forEach((h, idx) => {
          row[h] = parts[idx];
        });
        timeSeries.push(row);
      }
    } else {
      // Parsing Final Session Report Sections
      if (rawLine.startsWith("generated_at,")) {
        reportData.generatedAt = rawLine.split(",")[1]?.trim() || null;
        continue;
      }

      if (rawLine.startsWith("--") && rawLine.endsWith("--")) {
        const secTitle = rawLine.replace(/--/g, "").trim().toLowerCase();
        if (secTitle.includes("vehicle counts")) currentSection = "counts";
        else if (secTitle.includes("humans detected")) currentSection = "humans";
        else if (secTitle.includes("recognized license plates")) currentSection = "plates";
        else if (secTitle.includes("two-wheeler helmet")) currentSection = "twoWheelers";
        else if (secTitle.includes("per-vehicle")) currentSection = "perVehicle";
        else currentSection = null;
        continue;
      }

      // Process section lines
      const parts = parseCSVLine(rawLine);
      if (parts.length === 0) continue;

      if (currentSection === "counts") {
        if (parts[0].toLowerCase() === "vehicle_class") continue;
        const vcls = parts[0]?.trim();
        const cnt = parseInt(parts[1], 10) || 0;
        if (vcls) reportData.vehicleCounts[vcls] = cnt;
      } else if (currentSection === "humans") {
        if (parts[0].toLowerCase() === "category") continue;
        const cat = parts[0]?.trim().toLowerCase();
        const cnt = parseInt(parts[1], 10) || 0;
        if (cat) reportData.humans[cat] = cnt;
      } else if (currentSection === "plates") {
        if (parts[0].toLowerCase() === "track_id") continue;
        if (parts.length >= 3) {
          reportData.plates.push({
            track_id: parts[0]?.trim(),
            vehicle_class: parts[1]?.trim() || "Car",
            plate: parts[2]?.trim(),
            confidence: parts[3]?.trim() || "Confirmed"
          });
        }
      } else if (currentSection === "twoWheelers") {
        if (parts[0].toLowerCase() === "track_id") continue;
        if (parts.length >= 6) {
          reportData.twoWheelers.push({
            track_id: parts[0]?.trim(),
            vehicle_class: parts[1]?.trim() || "Bike/Motorcycle",
            plate: parts[2]?.trim() || "not detected",
            rider_helmet: parts[3]?.trim() || "Unknown",
            pillion_helmet: parts[4]?.trim() || "None",
            verdict: parts[5]?.trim() || "COMPLIANT"
          });
        }
      } else if (currentSection === "perVehicle") {
        if (parts[0].toLowerCase() === "track_id") continue;
        if (parts.length >= 4) {
          reportData.perVehicle.push({
            track_id: parts[0]?.trim(),
            vehicle_class: parts[1]?.trim(),
            plate: parts[2]?.trim(),
            helmet_status: parts[3]?.trim()
          });
        }
      }
    }
  }

  if (inFinalReport) {
    report = reportData;
  }

  return { timeSeries, report };
}

/**
 * Extracts comprehensive report data from /analytics row dictionaries
 * in case the user has not directly uploaded raw text.
 */
export function parseAnalyticsRows(rows = []) {
  if (!rows || rows.length === 0) return null;

  // Find the last row with valid vehicle numbers
  let lastRow = null;
  for (let i = rows.length - 1; i >= 0; i--) {
    if (rows[i] && rows[i].total_vehicles !== undefined) {
      lastRow = rows[i];
      break;
    }
  }
  if (!lastRow) return null;

  const vehicleCounts = {
    Car: Number(lastRow.cars) || 0,
    "Bike/Motorcycle": (Number(lastRow.motorcycles) || 0) + (Number(lastRow.scooters) || 0),
    "Auto Rickshaw": Number(lastRow.auto_rickshaws) || 0,
    Truck: Number(lastRow.trucks) || 0,
    Bus: Number(lastRow.buses) || 0,
    Bicycle: Number(lastRow.bicycles) || 0,
    total: Number(lastRow.total_vehicles) || 0
  };

  // Extract all unique plates seen across all rows
  const platesMap = new Map();
  rows.forEach(r => {
    if (r.plates_detected && r.plates_detected !== "none") {
      const tokens = String(r.plates_detected).split("|");
      tokens.forEach(tok => {
        const trimmed = tok.trim();
        if (trimmed && trimmed.includes(":")) {
          const [tid, plt] = trimmed.split(":");
          if (plt && !platesMap.has(plt)) {
            platesMap.set(plt, {
              track_id: tid,
              plate: plt,
              vehicle_class: "Car",
              confidence: "88.0%"
            });
          }
        }
      });
    }
  });

  // Extract all two-wheeler violations
  const violationsMap = new Map();
  rows.forEach(r => {
    if (r.violation_details && r.violation_details !== "none") {
      const tokens = String(r.violation_details).split("|");
      tokens.forEach(tok => {
        const trimmed = tok.trim();
        if (trimmed && trimmed.includes(":")) {
          const [tid, violType] = trimmed.split(":");
          if (!violationsMap.has(tid)) {
            violationsMap.set(tid, {
              track_id: tid,
              vehicle_class: "Bike/Motorcycle",
              plate: "not detected",
              rider_helmet: violType.includes("rider") ? "NO HELMET" : "HELMET",
              pillion_helmet: violType.includes("pillion") ? "NO HELMET" : "None",
              verdict: "❌ VIOLATION"
            });
          } else {
            const cur = violationsMap.get(tid);
            if (violType.includes("rider")) cur.rider_helmet = "NO HELMET";
            if (violType.includes("pillion")) cur.pillion_helmet = "NO HELMET";
          }
        }
      });
    }
  });

  // Demographic totals from last row
  const males = Number(lastRow.males) || 0;
  const females = Number(lastRow.females) || 0;
  const children = Number(lastRow.children) || 0;
  const pedTotal = Number(lastRow.pedestrians_detected) || (males + females + children);
  const unknown = Math.max(0, pedTotal - males - females - children);

  return {
    generatedAt: lastRow.timestamp || new Date().toISOString(),
    vehicleCounts,
    humans: { total: pedTotal, males, females, children, unknown },
    plates: Array.from(platesMap.values()),
    twoWheelers: Array.from(violationsMap.values()),
    perVehicle: []
  };
}

// Helper to handle commas inside quotes
function parseCSVLine(text) {
  const result = [];
  let cur = "";
  let inQuotes = false;

  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === '"') {
      inQuotes = !inQuotes;
    } else if (c === ',' && !inQuotes) {
      result.push(cur.trim());
      cur = "";
    } else {
      cur += c;
    }
  }
  result.push(cur.trim());
  return result;
}
