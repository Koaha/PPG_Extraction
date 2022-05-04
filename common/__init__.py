"""
vital_sqi.common
================
A subpackage for shared operations across package vital_sqi
"""

from common.generate_template import (
	ppg_dual_double_frequency_template,
	ppg_absolute_dual_skewness_template,
	ppg_nonlinear_dynamic_system_template,
	ecg_dynamic_template,
	)
from common.rpeak_detection import (
	PeakDetector
	)
from common.utils import *
