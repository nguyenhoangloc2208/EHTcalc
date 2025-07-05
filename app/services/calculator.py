from typing import List, Dict, Any
from app.models.equipment import Equipment
from app.models.hunter import Hunter
from itertools import combinations

class CalculatorService:
    @staticmethod
    def calculate_build_score(total_stats: Dict[str, float], weights: Dict[str, float]) -> float:
        """Calculate a score for a particular build based on weighted stats."""
        score = 0
        for stat, value in total_stats.items():
            if stat in weights:
                score += value * weights[stat]
        return score
    
    @staticmethod
    def find_optimal_build(
        hunter: Hunter,
        available_equipment: Dict[str, List[Equipment]],
        weights: Dict[str, float],
        max_items_per_type: Dict[str, int]
    ) -> Dict[str, Any]:
        """Find the optimal equipment combination for given stats weights."""
        best_score = 0
        best_build = None
        best_stats = None
        
        # Generate equipment combinations
        equipment_combinations = []
        for equip_type, max_items in max_items_per_type.items():
            if equip_type in available_equipment:
                items = available_equipment[equip_type]
                # Get all possible combinations of items for this type
                for r in range(min(max_items, len(items)) + 1):
                    equipment_combinations.extend(combinations(items, r))
        
        # Try each combination
        for combo in combinations(equipment_combinations, len(max_items_per_type)):
            # Flatten the combination and check for duplicates
            equipment_list = []
            for sublist in combo:
                equipment_list.extend(sublist)
            
            # Calculate total stats for this combination
            total_stats = hunter.calculate_total_stats(equipment_list)
            
            # Calculate score
            score = CalculatorService.calculate_build_score(total_stats, weights)
            
            # Update best build if score is higher
            if score > best_score:
                best_score = score
                best_build = equipment_list
                best_stats = total_stats
        
        return {
            'equipment': [e.to_dict() for e in best_build] if best_build else [],
            'total_stats': best_stats,
            'score': best_score
        }
    
    @staticmethod
    def compare_builds(
        hunter: Hunter,
        build1: List[Equipment],
        build2: List[Equipment]
    ) -> Dict[str, Any]:
        """Compare two different equipment builds."""
        stats1 = hunter.calculate_total_stats(build1)
        stats2 = hunter.calculate_total_stats(build2)
        
        # Calculate differences
        diff = {}
        for stat in stats1.keys():
            diff[stat] = stats2[stat] - stats1[stat]
        
        return {
            'build1_stats': stats1,
            'build2_stats': stats2,
            'differences': diff
        } 