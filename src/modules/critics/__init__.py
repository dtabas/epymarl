from .coma import COMACritic
from .centralV import CentralVCritic
from .coma_ns import COMACriticNS
from .centralV_ns import CentralVCriticNS
from .maddpg import MADDPGCritic
from .maddpg_ns import MADDPGCriticNS
from .ac import ACCritic
from .ac_ns import ACCriticNS
from .maddpg_pd_ns import MADDPGPDCriticNS
from .centralV_pd_ns import CentralVCriticPDNS
from .centralV_pd_ns_continuous import CentralVCriticPDNSContinuous
REGISTRY = {}

REGISTRY["coma_critic"] = COMACritic
REGISTRY["cv_critic"] = CentralVCritic
REGISTRY["coma_critic_ns"] = COMACriticNS
REGISTRY["cv_critic_ns"] = CentralVCriticNS
REGISTRY["maddpg_critic"] = MADDPGCritic
REGISTRY["maddpg_critic_ns"] = MADDPGCriticNS
REGISTRY["ac_critic"] = ACCritic
REGISTRY["ac_critic_ns"] = ACCriticNS
REGISTRY["maddpg_pd_critic_ns"] = MADDPGPDCriticNS
REGISTRY["cv_pd_critic_ns"] = CentralVCriticPDNS
REGISTRY["centralV_pd_ns_continuous"] = CentralVCriticPDNSContinuous