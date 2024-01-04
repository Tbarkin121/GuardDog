# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#aARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from .anymal import Anymal
from .anymal_terrain import AnymalTerrain
from .dynasoar import Dynasoar
from .dynasoarui import DynasoarUI
from .dynasoar2 import Dynasoar2
from .torquepole import TorquePole

# Mappings from strings to environments
isaacgym_task_map = {
    "Anymal": Anymal,
    "AnymalTerrain": AnymalTerrain,
    "Dynasoar": Dynasoar,
    "A_Wing": Dynasoar,
    "DynasoarUI": DynasoarUI,
    "Dynasoar2": Dynasoar2,
    "TorquePole": TorquePole,
}