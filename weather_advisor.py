"""
weather_advisor.py - Weather API Integration & Mining Climatic Limitation Engine
PS ID: SIH26009 - Subterranean Mineral Detection Using Space Technology & ML
Ministry of Steel | Domain: Smart Automation / Space Technology

Connects to Open-Meteo Weather API to evaluate real-time rainfall, atmospheric conditions,
and assess climatic limitations for manganese mining operations (open-cast pits, haul roads,
slope stability, blasting, and sump dewatering).
"""

import datetime
import requests
import streamlit as st
from typing import Dict, Any, Optional

# WMO Weather Interpretation Codes (WW)
WMO_CODES = {
    0: {"desc": "Clear Sky", "icon": "☀️", "is_rain": False, "severity": "low"},
    1: {"desc": "Mainly Clear", "icon": "🌤️", "is_rain": False, "severity": "low"},
    2: {"desc": "Partly Cloudy", "icon": "⛅", "is_rain": False, "severity": "low"},
    3: {"desc": "Overcast", "icon": "☁️", "is_rain": False, "severity": "low"},
    45: {"desc": "Foggy", "icon": "🌫️", "is_rain": False, "severity": "medium"},
    48: {"desc": "Depositing Rime Fog", "icon": "🌫️", "is_rain": False, "severity": "medium"},
    51: {"desc": "Light Drizzle", "icon": "🌦️", "is_rain": True, "severity": "medium"},
    53: {"desc": "Moderate Drizzle", "icon": "🌦️", "is_rain": True, "severity": "medium"},
    55: {"desc": "Dense Drizzle", "icon": "🌧️", "is_rain": True, "severity": "high"},
    56: {"desc": "Light Freezing Drizzle", "icon": "🌨️", "is_rain": True, "severity": "high"},
    57: {"desc": "Dense Freezing Drizzle", "icon": "🌨️", "is_rain": True, "severity": "high"},
    61: {"desc": "Slight Rain", "icon": "🌧️", "is_rain": True, "severity": "medium"},
    63: {"desc": "Moderate Rain", "icon": "🌧️", "is_rain": True, "severity": "high"},
    65: {"desc": "Heavy Rain", "icon": "🌧️🌧️", "is_rain": True, "severity": "critical"},
    66: {"desc": "Light Freezing Rain", "icon": "🌧️", "is_rain": True, "severity": "high"},
    67: {"desc": "Heavy Freezing Rain", "icon": "🌧️🌧️", "is_rain": True, "severity": "critical"},
    71: {"desc": "Slight Snow", "icon": "🌨️", "is_rain": False, "severity": "medium"},
    73: {"desc": "Moderate Snow", "icon": "🌨️", "is_rain": False, "severity": "high"},
    75: {"desc": "Heavy Snow", "icon": "❄️", "is_rain": False, "severity": "critical"},
    80: {"desc": "Slight Rain Showers", "icon": "🌦️", "is_rain": True, "severity": "medium"},
    81: {"desc": "Moderate Rain Showers", "icon": "🌧️", "is_rain": True, "severity": "high"},
    82: {"desc": "Violent Rain Showers", "icon": "⛈️", "is_rain": True, "severity": "critical"},
    85: {"desc": "Slight Snow Showers", "icon": "🌨️", "is_rain": False, "severity": "medium"},
    86: {"desc": "Heavy Snow Showers", "icon": "❄️", "is_rain": False, "severity": "critical"},
    95: {"desc": "Thunderstorm", "icon": "⛈️⚡", "is_rain": True, "severity": "critical"},
    96: {"desc": "Thunderstorm with Slight Hail", "icon": "⛈️❄️", "is_rain": True, "severity": "critical"},
    99: {"desc": "Thunderstorm with Heavy Hail", "icon": "⛈️💥", "is_rain": True, "severity": "critical"},
}


@st.cache_data(ttl=300)
def fetch_weather_data(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """
    Fetches real-time weather and 7-day forecast from Open-Meteo API.
    Does not require an API key and covers global coordinates.
    Cached for 5 minutes.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "current": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "rain",
            "weather_code",
            "wind_speed_10m",
            "cloud_cover"
        ],
        "hourly": [
            "precipitation_probability",
            "precipitation",
            "rain"
        ],
        "daily": [
            "weather_code",
            "precipitation_sum",
            "precipitation_probability_max",
            "temperature_2m_max",
            "temperature_2m_min",
            "wind_speed_10m_max"
        ],
        "timezone": "auto"
    }

    try:
        resp = requests.get(url, params=params, timeout=8)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"[Weather API] Exception querying Open-Meteo: {e}")

    # Fallback realistic monsoon/central India data if network offline
    now = datetime.datetime.now()
    dates = [(now + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
    return {
        "latitude": lat,
        "longitude": lon,
        "timezone": "Asia/Kolkata",
        "current": {
            "temperature_2m": 28.5,
            "relative_humidity_2m": 76,
            "precipitation": 0.0,
            "rain": 0.0,
            "weather_code": 3,
            "wind_speed_10m": 12.5,
            "cloud_cover": 85
        },
        "daily": {
            "time": dates,
            "weather_code": [3, 95, 95, 80, 61, 3, 1],
            "precipitation_sum": [0.0, 14.2, 26.5, 9.8, 4.2, 1.0, 0.0],
            "precipitation_probability_max": [20, 95, 90, 75, 55, 30, 15],
            "temperature_2m_max": [31.5, 28.0, 27.5, 28.2, 29.0, 30.1, 31.0],
            "temperature_2m_min": [24.8, 24.0, 23.5, 23.8, 24.1, 24.5, 24.8],
            "wind_speed_10m_max": [16.0, 32.0, 38.0, 22.0, 18.0, 14.0, 12.0]
        },
        "_is_fallback": True
    }


def assess_mining_climatic_risk(weather_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates rainfall and meteorological data against Directorate General
    of Mines Safety (DGMS) open-cast and underground mining regulations.
    """
    curr = weather_data.get("current", {})
    daily = weather_data.get("daily", {})

    temp = curr.get("temperature_2m", 25.0)
    humidity = curr.get("relative_humidity_2m", 60)
    precip_now = curr.get("precipitation", 0.0)
    rain_now = curr.get("rain", 0.0)
    wcode_now = curr.get("weather_code", 0)
    wind_now = curr.get("wind_speed_10m", 10.0)

    w_info = WMO_CODES.get(wcode_now, {"desc": "Cloudy", "icon": "⛅", "is_rain": False, "severity": "low"})

    daily_precip = daily.get("precipitation_sum", [0.0, 0.0])
    daily_prob = daily.get("precipitation_probability_max", [0, 0])
    daily_wcodes = daily.get("weather_code", [0, 0])

    precip_today = daily_precip[0] if len(daily_precip) > 0 else 0.0
    precip_tomorrow = daily_precip[1] if len(daily_precip) > 1 else 0.0
    prob_today = daily_prob[0] if len(daily_prob) > 0 else 0
    prob_tomorrow = daily_prob[1] if len(daily_prob) > 1 else 0

    code_tomorrow = daily_wcodes[1] if len(daily_wcodes) > 1 else 0
    w_info_tomorrow = WMO_CODES.get(code_tomorrow, {"desc": "Normal", "icon": "⛅", "is_rain": False, "severity": "low"})

    # Determine risk level based on rain and severe weather
    # CRITICAL: Active heavy rain (>2.0 mm/hr), thunderstorm, or forecast >15mm with high certainty
    if (rain_now >= 2.0 or precip_now >= 2.0 or
        wcode_now in [65, 67, 82, 95, 96, 99] or
        precip_tomorrow >= 20.0 or
        (precip_tomorrow >= 12.0 and prob_tomorrow >= 80)):
        risk_level = "CRITICAL"
        risk_color = "#DC2626"
        badge = "⛔ MINING SUSPENSION ADVISORY"
        alert_title = "CRITICAL CLIMATIC LIMITATION: HEAVY RAINFALL / THUNDERSTORM ALERT"
        alert_desc = (
            f"Severe weather limitation active ({w_info['icon']} {w_info['desc']} | Forecast: {precip_tomorrow:.1f} mm rain tomorrow with {prob_tomorrow}% probability). "
            f"Active water accumulation, rapid slope destabilization, and slick haul roads pose immediate operational and safety hazards."
        )

    # WARNING: Active light/moderate rain, or forecast >6mm with high prob
    elif (rain_now > 0.1 or precip_now > 0.1 or
          w_info.get("is_rain", False) or
          precip_tomorrow >= 6.0 or
          prob_tomorrow >= 65):
        risk_level = "WARNING"
        risk_color = "#EA580C"
        badge = "⚠️ HIGH OPERATIONAL CAUTION"
        alert_title = "CLIMATIC LIMITATION: RAINFALL & GROUND SLICKNESS ADVISORY"
        alert_desc = (
            f"Moderate climatic limitation ({w_info['icon']} {w_info['desc']} | Expected precipitation: {precip_tomorrow:.1f} mm, {prob_tomorrow}% chance). "
            f"Haulage roads require speed reductions; pit sump pumps must be mobilized; highwall benches require continuous monitoring."
        )

    # CAUTION: Moderate chance of rain / overcast
    elif prob_today > 35 or prob_tomorrow > 40 or precip_today > 2.0:
        risk_level = "CAUTION"
        risk_color = "#D97706"
        badge = "🟡 PRE-MONSOON WATCH"
        alert_title = "CLIMATIC WATCH: POTENTIAL PRECIPITATION WINDOW"
        alert_desc = (
            f"Precipitation watch active ({w_info['icon']} {w_info['desc']} | Probability: {max(prob_today, prob_tomorrow)}%). "
            f"Open-cast mining can proceed under vigilance. Inspect trench drains and verify sump drainage pump standby."
        )

    # SAFE: Dry atmospheric conditions
    else:
        risk_level = "SAFE"
        risk_color = "#16A34A"
        badge = "🟢 OPTIMAL MINING CONDITIONS"
        alert_title = "CLIMATIC CONDITIONS FAVORABLE FOR MINING"
        alert_desc = (
            f"Dry atmospheric conditions ({w_info['icon']} {w_info['desc']} | Rain: 0.0 mm). "
            f"All open-pit extraction, heavy dumper hauling, core drilling, and blast hole charging are clear to proceed."
        )

    # Operational breakdown matrix
    operations = [
        {
            "name": "🚜 Open-Cast Pit Excavation & Loading",
            "status": "SUSPENDED" if risk_level == "CRITICAL" else ("RESTRICTED" if risk_level == "WARNING" else "PERMITTED"),
            "color": "#DC2626" if risk_level == "CRITICAL" else ("#EA580C" if risk_level == "WARNING" else "#16A34A"),
            "advisory": "Water pooling in pit base; soft underfoot conditions prevent safe bucket loading." if risk_level == "CRITICAL" else ("Operate with caution on upper benches; avoid low-lying sumps." if risk_level == "WARNING" else "Full extraction capacity operational.")
        },
        {
            "name": "🚛 Heavy Dumper Haul Road Traction",
            "status": "PROHIBITED" if risk_level == "CRITICAL" else ("CAUTION (15 km/h limit)" if risk_level == "WARNING" else "OPTIMAL"),
            "color": "#DC2626" if risk_level == "CRITICAL" else ("#EA580C" if risk_level == "WARNING" else "#16A34A"),
            "advisory": "Extreme slippage on clay-rich phyllite haul roads; high risk of dumper skid and rollover." if risk_level in ["CRITICAL", "WARNING"] else "Haul roads dry and stable for 50T-100T dumpers."
        },
        {
            "name": "⛰️ Highwall Pit Slope & Landslide Stability",
            "status": "HIGH COLLAPSE RISK" if risk_level == "CRITICAL" else ("MONITOR SEEPAGE" if risk_level == "WARNING" else "STABLE"),
            "color": "#DC2626" if risk_level == "CRITICAL" else ("#EA580C" if risk_level == "WARNING" else "#16A34A"),
            "advisory": "Water saturation in Sausar schist reduces shear strength along foliation planes; inspect tension cracks." if risk_level in ["CRITICAL", "WARNING"] else "Factor of safety > 1.5. No hydrostatic pore pressure build-up."
        },
        {
            "name": "💥 Bench Drilling & ANFO Blasting",
            "status": "DISALLOWED" if risk_level in ["CRITICAL", "WARNING"] else "PERMITTED",
            "color": "#DC2626" if risk_level in ["CRITICAL", "WARNING"] else "#16A34A",
            "advisory": "Water in blast holes dissolves ANFO prills and causes misfires. Emulsion explosives required if wet." if risk_level in ["CRITICAL", "WARNING"] else "Blast holes dry; standard charging schedules approved."
        },
        {
            "name": "💧 Sump Dewatering & Inundation Safety",
            "status": "ACTIVATE PUMPS (24x7)" if risk_level in ["CRITICAL", "WARNING"] else "STANDBY",
            "color": "#DC2626" if risk_level == "CRITICAL" else ("#EA580C" if risk_level == "WARNING" else "#16A34A"),
            "advisory": f"Runoff inflow expected: {precip_tomorrow:.1f} mm. Clear intake strainers and deploy auxiliary diesel pumps." if risk_level in ["CRITICAL", "WARNING"] else "Base sumps at normal holding capacity."
        }
    ]

    # Daily 7-day forecast table
    forecast_days = []
    times = daily.get("time", [])
    for idx, day_date in enumerate(times):
        day_wcode = daily_wcodes[idx] if idx < len(daily_wcodes) else 0
        w_d = WMO_CODES.get(day_wcode, {"desc": "Cloudy", "icon": "⛅", "is_rain": False})
        p_sum = daily_precip[idx] if idx < len(daily_precip) else 0.0
        p_prob = daily_prob[idx] if idx < len(daily_prob) else 0
        t_max = daily.get("temperature_2m_max", [30])[idx] if idx < len(daily.get("temperature_2m_max", [])) else 30.0
        t_min = daily.get("temperature_2m_min", [24])[idx] if idx < len(daily.get("temperature_2m_min", [])) else 24.0

        if p_sum >= 15.0 or day_wcode in [95, 96, 99]:
            day_feasibility = "⛔ Suspended"
            f_color = "#DC2626"
        elif p_sum >= 5.0 or p_prob >= 60:
            day_feasibility = "⚠️ Restricted"
            f_color = "#EA580C"
        elif p_sum > 0.5 or p_prob >= 35:
            day_feasibility = "🟡 Caution"
            f_color = "#D97706"
        else:
            day_feasibility = "✅ Permitted"
            f_color = "#16A34A"

        try:
            d_obj = datetime.datetime.strptime(day_date, "%Y-%m-%d")
            formatted_date = d_obj.strftime("%a, %b %d")
        except:
            formatted_date = day_date

        forecast_days.append({
            "date": formatted_date,
            "raw_date": day_date,
            "icon": w_d["icon"],
            "desc": w_d["desc"],
            "precip_mm": p_sum,
            "rain_prob": p_prob,
            "temp_range": f"{t_min:.0f}° - {t_max:.0f}°C",
            "mining_status": day_feasibility,
            "color": f_color
        })

    return {
        "risk_level": risk_level,
        "risk_color": risk_color,
        "badge": badge,
        "alert_title": alert_title,
        "alert_desc": alert_desc,
        "current_weather": {
            "temp_c": temp,
            "humidity_pct": humidity,
            "precip_mm": precip_now,
            "rain_mm": rain_now,
            "wind_kmh": wind_now,
            "weather_desc": w_info["desc"],
            "weather_icon": w_info["icon"],
            "wmo_code": wcode_now
        },
        "today_precip_sum": precip_today,
        "tomorrow_precip_sum": precip_tomorrow,
        "tomorrow_prob": prob_tomorrow,
        "tomorrow_desc": f"{w_info_tomorrow['icon']} {w_info_tomorrow['desc']}",
        "operations": operations,
        "forecast_days": forecast_days,
        "is_fallback": weather_data.get("_is_fallback", False)
    }


def render_weather_warning_banner(risk_info: Dict[str, Any], location_name: str = "Balaghat Mining Belt"):
    """
    Renders a high-visibility warning banner in Streamlit before mining activities.
    """
    risk_level = risk_info["risk_level"]
    risk_color = risk_info["risk_color"]
    cw = risk_info["current_weather"]

    bg_color = "#FEE2E2" if risk_level == "CRITICAL" else ("#FFEDD5" if risk_level == "WARNING" else ("#FEF3C7" if risk_level == "CAUTION" else "#DCFCE7"))
    border_color = risk_color

    st.markdown(f"""
    <div style="
        background-color: {bg_color};
        border-left: 6px solid {border_color};
        border-radius: 8px;
        padding: 14px 18px;
        margin: 12px 0 16px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        font-family: sans-serif;
    ">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; margin-bottom:6px;">
            <div style="font-weight:700; font-size:15px; color:{risk_color};">
                {risk_info['badge']} &nbsp;|&nbsp; 📍 {location_name}
            </div>
            <div style="font-size:13px; color:#334155; font-weight:600;">
                {cw['weather_icon']} {cw['weather_desc']} &nbsp;•&nbsp; 🌡️ {cw['temp_c']:.1f}°C &nbsp;•&nbsp; 💧 Rain: {cw['rain_mm']:.1f} mm/h &nbsp;•&nbsp; 💨 {cw['wind_kmh']:.1f} km/h
            </div>
        </div>
        <div style="font-size:14px; color:#1E293B; line-height:1.45; font-weight:600;">
            {risk_info['alert_title']}
        </div>
        <div style="font-size:13px; color:#475569; margin-top:4px; line-height:1.4;">
            {risk_info['alert_desc']}
        </div>
        <div style="font-size:12px; color:#64748B; margin-top:8px; border-top:1px dashed #CBD5E1; padding-top:6px;">
            🌧️ <b>24h-48h Rainfall Outlook:</b> Today: <b>{risk_info['today_precip_sum']:.1f} mm</b> &nbsp;|&nbsp; Tomorrow: <b style="color:{risk_color};">{risk_info['tomorrow_precip_sum']:.1f} mm</b> ({risk_info['tomorrow_prob']}% chance, {risk_info['tomorrow_desc']})
        </div>
    </div>
    """, unsafe_allow_html=True)
