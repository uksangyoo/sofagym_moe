import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

import sofagym_moe

print('sofagym_moe version:', sofagym_moe.__version__)
sofagym_moe.register.register_all()
print('Registered MOE env subpackages.')
