import sys
from pathlib import Path

# Allow importing sibling packages (e.g., impls.saw) when running from impls/.
REPO_ROOT = Path(__file__).resolve().parents[2]
IMPLS_DIR = str(REPO_ROOT / "impls")
if IMPLS_DIR not in sys.path:
    sys.path.append(IMPLS_DIR)

from agents.crl import CRLAgent
from agents.gcbc import GCBCAgent
from agents.gciql import GCIQLAgent
from agents.gcivl import GCIVLAgent
from agents.hiql import HIQLAgent
from agents.ota import OTAAgent
from agents.qrl import QRLAgent
from agents.qrl_simpl import QRLSimplAgent
from agents.sac import SACAgent
from agents.saw import SAWAgent

agents = dict(
    crl=CRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gcivl=GCIVLAgent,
    hiql=HIQLAgent,
    ota=OTAAgent,
    qrl=QRLAgent,
    sac=SACAgent,
    qrl_simpl=QRLSimplAgent,
    saw=SAWAgent,
)
