"""
Advanced Safety Systems for LQG Positive Matter Assembler

This module implements comprehensive safety protocols including emergency
termination systems, biological protection measures, containment protocols,
and real-time hazard assessment for the LQG positive matter assembler.

Key Features:
- 10¹² biological protection margin with <1μs emergency response
- Multi-layer safety validation with redundant protection systems  
- Real-time hazard assessment and risk mitigation
- Automated emergency termination on constraint violations
- Comprehensive safety logging and incident reporting
"""

import numpy as np
import time
import logging
import threading
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SafetyLevel(Enum):
    """Safety assessment levels"""
    SAFE = "SAFE"
    CAUTION = "CAUTION"
    WARNING = "WARNING"
    DANGER = "DANGER"
    EMERGENCY = "EMERGENCY"

class HazardType(Enum):
    """Types of hazards to monitor"""
    ENERGY_CONDITION_VIOLATION = "energy_condition_violation"
    EXCESSIVE_ENERGY_DENSITY = "excessive_energy_density"
    UNCONTROLLED_MATTER_ASSEMBLY = "uncontrolled_matter_assembly"
    CAUSALITY_VIOLATION = "causality_violation"
    CONTAINMENT_BREACH = "containment_breach"
    BIOLOGICAL_HAZARD = "biological_hazard"
    SYSTEM_MALFUNCTION = "system_malfunction"

@dataclass
class SafetyThresholds:
    """Safety threshold configuration"""
    # Energy density limits (J/m³)
    max_energy_density: float = 1e18
    max_energy_density_gradient: float = 1e17  # Per meter
    
    # Biological protection
    biological_protection_factor: float = 1e12
    max_biological_exposure: float = 1e-12  # Fraction of safe limit
    
    # Temporal constraints
    emergency_response_time: float = 1e-6  # 1 microsecond
    max_assembly_duration: float = 3600.0  # 1 hour
    
    # Spatial constraints
    max_assembly_radius: float = 1000.0  # 1 km
    containment_safety_margin: float = 10.0  # 10× safety margin
    
    # Conservation tolerances
    max_conservation_error: float = 0.001  # 0.1%
    max_energy_flux_divergence: float = 1e-6
    
    # System performance
    min_system_health: float = 0.95  # 95% minimum health
    max_constraint_violations: int = 3

@dataclass
class SafetyIncident:
    """Record of safety incident"""
    timestamp: float
    incident_id: str
    hazard_type: HazardType
    safety_level: SafetyLevel
    location: Tuple[float, float, float]
    description: str
    affected_systems: List[str]
    response_actions: List[str]
    resolution_time: Optional[float] = None
    lessons_learned: Optional[str] = None

@dataclass
class BiologicalProtectionStatus:
    """Status of biological protection systems"""
    protection_factor: float
    exposure_level: float
    safe_distance: float
    radiation_shielding_active: bool
    containment_integrity: float
    emergency_evacuation_ready: bool

class EmergencyTerminationSystem:
    """
    Ultra-fast emergency termination system with <1μs response time
    """
    
    def __init__(self, response_time_target: float = 1e-6):
        """
        Initialize emergency termination system
        
        Args:
            response_time_target: Target response time in seconds
        """
        self.response_time_target = response_time_target
        self.armed = False
        self.termination_callbacks = []
        self.last_activation_time = None
        self.activation_count = 0
        
        # Hardware-level termination flags
        self._hardware_termination_flag = threading.Event()
        self._software_termination_flag = threading.Event()
        
        logger.info(f"Emergency termination system initialized (target: {response_time_target*1e6:.1f} μs)")
    
    def arm_system(self):
        """Arm the emergency termination system"""
        self.armed = True
        self._hardware_termination_flag.clear()
        self._software_termination_flag.clear()
        logger.info("Emergency termination system ARMED")
    
    def disarm_system(self):
        """Disarm the emergency termination system"""
        self.armed = False
        logger.info("Emergency termination system DISARMED")
    
    def register_termination_callback(self, callback: Callable[[str, Any], None]):
        """
        Register callback for emergency termination
        
        Args:
            callback: Function to call on emergency termination
        """
        self.termination_callbacks.append(callback)
        logger.info(f"Registered emergency termination callback: {callback.__name__}")
    
    def trigger_emergency_termination(self, reason: str, context: Any = None):
        """
        Trigger immediate emergency termination
        
        Args:
            reason: Reason for emergency termination
            context: Additional context information
        """
        if not self.armed:
            logger.warning("Emergency termination triggered but system not armed")
            return
        
        activation_start = time.time()
        self.last_activation_time = activation_start
        self.activation_count += 1
        
        logger.critical(f"EMERGENCY TERMINATION ACTIVATED: {reason}")
        
        # Set hardware termination flag (fastest response)
        self._hardware_termination_flag.set()
        
        # Execute termination callbacks
        for callback in self.termination_callbacks:
            try:
                callback(reason, context)
            except Exception as e:
                logger.error(f"Termination callback error: {e}")
        
        # Set software termination flag
        self._software_termination_flag.set()
        
        # Calculate response time
        response_time = time.time() - activation_start
        
        if response_time <= self.response_time_target:
            logger.info(f"Emergency termination completed in {response_time*1e6:.2f} μs (✅ TARGET MET)")
        else:
            logger.warning(f"Emergency termination took {response_time*1e6:.2f} μs (⚠️ EXCEEDED TARGET)")
        
        # Disarm system after activation
        self.armed = False
    
    def is_terminated(self) -> bool:
        """Check if emergency termination is active"""
        return self._hardware_termination_flag.is_set() or self._software_termination_flag.is_set()
    
    def get_termination_status(self) -> Dict[str, Any]:
        """Get termination system status"""
        return {
            'armed': self.armed,
            'terminated': self.is_terminated(),
            'activation_count': self.activation_count,
            'last_activation_time': self.last_activation_time,
            'response_time_target': self.response_time_target
        }

class BiologicalProtectionSystem:
    """
    Biological protection system with 10¹² safety margin
    """
    
    def __init__(self, protection_factor: float = 1e12):
        """
        Initialize biological protection system
        
        Args:
            protection_factor: Safety margin factor (default: 10¹²)
        """
        self.protection_factor = protection_factor
        self.base_safe_limit = 1e-3  # Base safe exposure limit (arbitrary units)
        self.current_exposure = 0.0
        self.monitoring_active = False
        
        # Shielding parameters
        self.radiation_shielding_active = True
        self.containment_fields_active = True
        self.safe_distance = 100.0  # 100m default safe distance
        
        logger.info(f"Biological protection system initialized (factor: {protection_factor:.0e})")
    
    def calculate_safe_exposure_limit(self) -> float:
        """Calculate safe exposure limit with protection factor"""
        return self.base_safe_limit / self.protection_factor
    
    def update_exposure_level(self, energy_density: float, distance: float) -> float:
        """
        Update current biological exposure level
        
        Args:
            energy_density: Energy density at source (J/m³)
            distance: Distance from source (m)
            
        Returns:
            Current exposure level
        """
        # Simple inverse square law approximation
        exposure_at_distance = energy_density / (4 * np.pi * distance**2 * 299792458.0**2)
        
        # Apply shielding factors
        if self.radiation_shielding_active:
            exposure_at_distance *= 1e-6  # 10⁶× shielding reduction
        
        if self.containment_fields_active:
            exposure_at_distance *= 1e-3  # 10³× containment reduction
        
        self.current_exposure = exposure_at_distance
        return self.current_exposure
    
    def assess_biological_safety(self, energy_density: float, distance: float) -> BiologicalProtectionStatus:
        """
        Assess current biological safety status
        
        Args:
            energy_density: Energy density at source (J/m³)
            distance: Distance from personnel (m)
            
        Returns:
            Biological protection status
        """
        current_exposure = self.update_exposure_level(energy_density, distance)
        safe_limit = self.calculate_safe_exposure_limit()
        
        # Calculate required safe distance
        required_distance = np.sqrt(energy_density / (4 * np.pi * safe_limit * 299792458.0**2))
        if self.radiation_shielding_active:
            required_distance *= 1e-3  # Reduced by shielding
        
        # Containment integrity (simplified)
        containment_integrity = 1.0 if self.containment_fields_active else 0.5
        
        return BiologicalProtectionStatus(
            protection_factor=self.protection_factor,
            exposure_level=current_exposure / safe_limit,  # Fraction of safe limit
            safe_distance=max(required_distance, self.safe_distance),
            radiation_shielding_active=self.radiation_shielding_active,
            containment_integrity=containment_integrity,
            emergency_evacuation_ready=True
        )
    
    def check_safety_compliance(self, energy_density: float, personnel_distance: float) -> Tuple[bool, str]:
        """
        Check if current conditions comply with biological safety requirements
        
        Args:
            energy_density: Current energy density (J/m³)
            personnel_distance: Distance to nearest personnel (m)
            
        Returns:
            (is_safe, safety_message)
        """
        status = self.assess_biological_safety(energy_density, personnel_distance)
        
        if status.exposure_level > 1.0:
            return False, f"Exposure exceeds safe limit by {status.exposure_level:.1f}×"
        elif status.exposure_level > 0.1:
            return False, f"Exposure at {status.exposure_level:.1%} of safe limit (>10% threshold)"
        elif personnel_distance < status.safe_distance:
            return False, f"Personnel too close: {personnel_distance:.1f}m < {status.safe_distance:.1f}m required"
        elif not status.radiation_shielding_active:
            return False, "Radiation shielding not active"
        elif status.containment_integrity < 0.9:
            return False, f"Containment integrity compromised: {status.containment_integrity:.1%}"
        else:
            return True, f"Biological safety compliant (exposure: {status.exposure_level:.2%})"

class ComprehensiveSafetySystem:
    """
    Comprehensive safety system integrating all protection measures
    """
    
    def __init__(self, safety_thresholds: Optional[SafetyThresholds] = None):
        """
        Initialize comprehensive safety system
        
        Args:
            safety_thresholds: Safety threshold configuration
        """
        self.thresholds = safety_thresholds or SafetyThresholds()
        
        # Initialize subsystems
        self.emergency_system = EmergencyTerminationSystem(
            response_time_target=self.thresholds.emergency_response_time
        )
        self.biological_protection = BiologicalProtectionSystem(
            protection_factor=self.thresholds.biological_protection_factor
        )
        
        # Safety monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.safety_incidents = []
        self.last_safety_assessment = None
        
        # Current system state
        self.current_energy_density = 0.0
        self.current_assembly_radius = 0.0
        self.personnel_distance = 1000.0  # 1km default safe distance
        self.system_health = 1.0
        
        # Register emergency callbacks
        self.emergency_system.register_termination_callback(self._emergency_matter_assembly_stop)
        self.emergency_system.register_termination_callback(self._emergency_containment_activation)
        
        logger.info("Comprehensive safety system initialized")
    
    def start_safety_monitoring(self):
        """Start continuous safety monitoring"""
        if self.monitoring_active:
            logger.warning("Safety monitoring already active")
            return
        
        self.monitoring_active = True
        self.emergency_system.arm_system()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._safety_monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Safety monitoring started")
    
    def stop_safety_monitoring(self):
        """Stop safety monitoring"""
        self.monitoring_active = False
        self.emergency_system.disarm_system()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        
        logger.info("Safety monitoring stopped")
    
    def update_system_state(self, energy_density: float, assembly_radius: float, 
                          personnel_distance: float, system_health: float):
        """
        Update current system state for safety assessment
        
        Args:
            energy_density: Current energy density (J/m³)
            assembly_radius: Current assembly radius (m)
            personnel_distance: Distance to nearest personnel (m)
            system_health: System health fraction (0.0 to 1.0)
        """
        self.current_energy_density = energy_density
        self.current_assembly_radius = assembly_radius
        self.personnel_distance = personnel_distance
        self.system_health = system_health
    
    def assess_safety_level(self) -> Tuple[SafetyLevel, List[str]]:
        """
        Assess current overall safety level
        
        Returns:
            (safety_level, warnings)
        """
        warnings = []
        max_safety_level = SafetyLevel.SAFE
        
        # Check energy density limits
        if self.current_energy_density > self.thresholds.max_energy_density:
            warnings.append(f"Energy density exceeds limit: {self.current_energy_density:.2e} > {self.thresholds.max_energy_density:.2e}")
            max_safety_level = SafetyLevel.EMERGENCY
        elif self.current_energy_density > 0.8 * self.thresholds.max_energy_density:
            warnings.append("Energy density approaching limit")
            max_safety_level = max(max_safety_level, SafetyLevel.WARNING)
        
        # Check assembly radius
        if self.current_assembly_radius > self.thresholds.max_assembly_radius:
            warnings.append(f"Assembly radius exceeds limit: {self.current_assembly_radius:.1f} > {self.thresholds.max_assembly_radius:.1f}")
            max_safety_level = SafetyLevel.DANGER
        
        # Check biological safety
        bio_safe, bio_message = self.biological_protection.check_safety_compliance(
            self.current_energy_density, self.personnel_distance
        )
        if not bio_safe:
            warnings.append(f"Biological safety: {bio_message}")
            max_safety_level = SafetyLevel.EMERGENCY
        
        # Check system health
        if self.system_health < self.thresholds.min_system_health:
            warnings.append(f"System health below minimum: {self.system_health:.1%} < {self.thresholds.min_system_health:.1%}")
            max_safety_level = max(max_safety_level, SafetyLevel.WARNING)
        
        return max_safety_level, warnings
    
    def check_emergency_conditions(self) -> bool:
        """
        Check if emergency termination should be triggered
        
        Returns:
            True if emergency termination required
        """
        safety_level, warnings = self.assess_safety_level()
        
        if safety_level == SafetyLevel.EMERGENCY:
            emergency_reason = f"Safety level EMERGENCY: {'; '.join(warnings)}"
            self.emergency_system.trigger_emergency_termination(
                emergency_reason, 
                context={
                    'energy_density': self.current_energy_density,
                    'assembly_radius': self.current_assembly_radius,
                    'personnel_distance': self.personnel_distance,
                    'system_health': self.system_health
                }
            )
            return True
        
        return False
    
    def _safety_monitoring_loop(self):
        """Main safety monitoring loop"""
        while self.monitoring_active:
            try:
                # Assess safety conditions
                safety_level, warnings = self.assess_safety_level()
                
                # Check for emergency conditions
                if safety_level == SafetyLevel.EMERGENCY:
                    self.check_emergency_conditions()
                    break
                
                # Log warnings if any
                if warnings and safety_level != SafetyLevel.SAFE:
                    logger.warning(f"Safety {safety_level.value}: {'; '.join(warnings)}")
                
                # Update last assessment
                self.last_safety_assessment = {
                    'timestamp': time.time(),
                    'safety_level': safety_level,
                    'warnings': warnings
                }
                
                # Sleep until next assessment (high frequency for safety)
                time.sleep(0.001)  # 1ms monitoring interval
                
            except Exception as e:
                logger.error(f"Safety monitoring error: {e}")
                self.emergency_system.trigger_emergency_termination(
                    f"Safety monitoring system error: {e}"
                )
                break
    
    def _emergency_matter_assembly_stop(self, reason: str, context: Any):
        """Emergency callback: Stop matter assembly immediately"""
        logger.critical("EMERGENCY: Stopping matter assembly")
        # This would interface with the matter assembler to halt operations
        # Implementation depends on specific matter assembler interface
    
    def _emergency_containment_activation(self, reason: str, context: Any):
        """Emergency callback: Activate maximum containment"""
        logger.critical("EMERGENCY: Activating maximum containment")
        self.biological_protection.containment_fields_active = True
        self.biological_protection.radiation_shielding_active = True
    
    def log_safety_incident(self, hazard_type: HazardType, description: str,
                          location: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                          affected_systems: Optional[List[str]] = None):
        """
        Log safety incident for analysis and reporting
        
        Args:
            hazard_type: Type of hazard detected
            description: Incident description
            location: Location coordinates where incident occurred
            affected_systems: List of affected system components
        """
        incident_id = f"SI_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.safety_incidents):03d}"
        
        incident = SafetyIncident(
            timestamp=time.time(),
            incident_id=incident_id,
            hazard_type=hazard_type,
            safety_level=self.assess_safety_level()[0],
            location=location,
            description=description,
            affected_systems=affected_systems or [],
            response_actions=[]
        )
        
        self.safety_incidents.append(incident)
        logger.warning(f"Safety incident logged: {incident_id} - {description}")
    
    def get_safety_status(self) -> Dict[str, Any]:
        """Get comprehensive safety status report"""
        safety_level, warnings = self.assess_safety_level()
        
        bio_status = self.biological_protection.assess_biological_safety(
            self.current_energy_density, self.personnel_distance
        )
        
        emergency_status = self.emergency_system.get_termination_status()
        
        return {
            'overall_safety_level': safety_level.value,
            'warnings': warnings,
            'biological_protection': {
                'protection_factor': bio_status.protection_factor,
                'exposure_level': bio_status.exposure_level,
                'safe_distance': bio_status.safe_distance,
                'containment_integrity': bio_status.containment_integrity
            },
            'emergency_system': emergency_status,
            'monitoring_active': self.monitoring_active,
            'recent_incidents': len([i for i in self.safety_incidents 
                                   if time.time() - i.timestamp < 3600]),  # Last hour
            'system_health': self.system_health,
            'last_assessment': self.last_safety_assessment
        }

def create_safety_system(biological_protection_factor: float = 1e12,
                        emergency_response_time: float = 1e-6,
                        max_energy_density: float = 1e18) -> ComprehensiveSafetySystem:
    """
    Factory function to create configured safety system
    
    Args:
        biological_protection_factor: Protection margin factor
        emergency_response_time: Emergency response time target (seconds)
        max_energy_density: Maximum allowed energy density (J/m³)
        
    Returns:
        Configured ComprehensiveSafetySystem instance
    """
    thresholds = SafetyThresholds(
        biological_protection_factor=biological_protection_factor,
        emergency_response_time=emergency_response_time,
        max_energy_density=max_energy_density
    )
    
    return ComprehensiveSafetySystem(thresholds)

# Example usage and demonstration
if __name__ == "__main__":
    print("🛡️ Comprehensive Safety System - Demonstration")
    print("=" * 60)
    
    # Create safety system
    safety_system = create_safety_system(
        biological_protection_factor=1e12,
        emergency_response_time=1e-6,
        max_energy_density=1e18
    )
    
    # Start monitoring
    safety_system.start_safety_monitoring()
    
    # Simulate normal operation
    print("✅ Normal operation - safe conditions")
    safety_system.update_system_state(
        energy_density=1e15,    # Safe level
        assembly_radius=10.0,   # 10m radius
        personnel_distance=100.0, # 100m distance
        system_health=0.98      # 98% health
    )
    
    time.sleep(0.1)
    status = safety_system.get_safety_status()
    print(f"Safety level: {status['overall_safety_level']}")
    
    # Simulate warning condition
    print("\n⚠️ Warning condition - elevated energy density")
    safety_system.update_system_state(
        energy_density=8e17,    # High but not emergency
        assembly_radius=10.0,
        personnel_distance=100.0,
        system_health=0.98
    )
    
    time.sleep(0.1)
    status = safety_system.get_safety_status()
    print(f"Safety level: {status['overall_safety_level']}")
    if status['warnings']:
        for warning in status['warnings']:
            print(f"  Warning: {warning}")
    
    # Test biological protection
    bio_status = safety_system.biological_protection.assess_biological_safety(1e15, 50.0)
    print(f"\n🧬 Biological protection status:")
    print(f"  Protection factor: {bio_status.protection_factor:.0e}")
    print(f"  Exposure level: {bio_status.exposure_level:.2%} of safe limit")
    print(f"  Required safe distance: {bio_status.safe_distance:.1f} m")
    
    # Stop monitoring
    safety_system.stop_safety_monitoring()
    print("\n🛑 Safety monitoring stopped")
    print("🎯 Safety system demonstration completed")
