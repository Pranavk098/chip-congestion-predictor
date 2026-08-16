"""Assembles the interactive demo artifact: injects base64 font data and
real exported prediction samples into the HTML template. Keeps the large
binary/data blobs out of the template file itself."""
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
TEMPLATE = Path(r"C:\Users\prana\AppData\Local\Temp\claude\C--Users-prana-OneDrive-Desktop-project-new\27db3404-73ae-413f-937e-bb2dcf851845\scratchpad\demo_template.html")
FONTS_DIR = PROJECT_DIR / "assets" / "fonts"
DATA_DIR = PROJECT_DIR / "assets" / "data"
OUT = Path(r"C:\Users\prana\AppData\Local\Temp\claude\C--Users-prana-OneDrive-Desktop-project-new\27db3404-73ae-413f-937e-bb2dcf851845\scratchpad\demo.html")

html = TEMPLATE.read_text(encoding="utf-8")

fonts = {
    "__PLEXSANS_REGULAR__": "PlexSans-Regular.b64.txt",
    "__PLEXSANS_SEMIBOLD__": "PlexSans-SemiBold.b64.txt",
    "__PLEXMONO_REGULAR__": "PlexMono-Regular.b64.txt",
    "__PLEXMONO_MEDIUM__": "PlexMono-Medium.b64.txt",
    "__PLEXMONO_SEMIBOLD__": "PlexMono-SemiBold.b64.txt",
}
for placeholder, fname in fonts.items():
    html = html.replace(placeholder, (FONTS_DIR / fname).read_text().strip())

metrics = json.loads((DATA_DIR / "metrics.json").read_text())
circuitnet = json.loads((DATA_DIR / "circuitnet_samples.json").read_text())
aes = json.loads((DATA_DIR / "aes_samples.json").read_text())

html = html.replace("__METRICS_JSON__", json.dumps(metrics))
html = html.replace("__CIRCUITNET_SAMPLES__", json.dumps(circuitnet))
html = html.replace("__AES_SAMPLES__", json.dumps(aes))
html = html.replace("__REPO_URL__", "https://github.com/pranavk-4/chip-congestion-predictor")

OUT.write_text(html, encoding="utf-8")
print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
