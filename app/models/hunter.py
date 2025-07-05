from app import db

class Hunter(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    class_type = db.Column(db.String(50), nullable=False)  # berserker, ranger, paladin, sorcerer
    
    # Base stats
    base_hp = db.Column(db.Integer, default=0)
    base_atk = db.Column(db.Integer, default=0)
    base_def = db.Column(db.Integer, default=0)
    base_crit = db.Column(db.Float, default=0)
    base_atk_spd = db.Column(db.Float, default=1.0)
    base_evasion = db.Column(db.Float, default=0)
    
    def calculate_total_stats(self, equipment_list):
        """Calculate total stats with equipped items."""
        total_stats = {
            'hp': self.base_hp,
            'atk': self.base_atk,
            'def': self.base_def,
            'crit': self.base_crit,
            'atk_spd': self.base_atk_spd,
            'evasion': self.base_evasion
        }
        
        # Add equipment stats
        for equip in equipment_list:
            total_stats['hp'] += equip.hp
            total_stats['atk'] += equip.atk
            total_stats['def'] += equip.def_
            total_stats['crit'] += equip.crit
            total_stats['atk_spd'] += equip.atk_spd
            total_stats['evasion'] += equip.evasion
        
        # Apply stat caps
        total_stats['crit'] = min(total_stats['crit'], 50.0)  # Max 50% crit
        total_stats['atk_spd'] = max(total_stats['atk_spd'], 0.25)  # Min 0.25 attack speed
        total_stats['evasion'] = min(total_stats['evasion'], 40.0)  # Max 40% evasion
        
        return total_stats
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'class_type': self.class_type,
            'base_stats': {
                'hp': self.base_hp,
                'atk': self.base_atk,
                'def': self.base_def,
                'crit': self.base_crit,
                'atk_spd': self.base_atk_spd,
                'evasion': self.base_evasion
            }
        }
    
    @staticmethod
    def from_dict(data):
        hunter = Hunter(
            name=data['name'],
            class_type=data['class_type'],
            base_hp=data['base_stats'].get('hp', 0),
            base_atk=data['base_stats'].get('atk', 0),
            base_def=data['base_stats'].get('def', 0),
            base_crit=data['base_stats'].get('crit', 0),
            base_atk_spd=data['base_stats'].get('atk_spd', 1.0),
            base_evasion=data['base_stats'].get('evasion', 0)
        )
        return hunter 