from app import db
from datetime import datetime

class Equipment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    type = db.Column(db.String(50), nullable=False)  # weapon, armor, accessory
    rarity = db.Column(db.String(20))  # normal, rare, epic, etc.
    
    # Base stats
    hp = db.Column(db.Integer, default=0)
    atk = db.Column(db.Integer, default=0)
    def_ = db.Column(db.Integer, default=0)  # defense
    crit = db.Column(db.Float, default=0)  # critical chance
    atk_spd = db.Column(db.Float, default=0)  # attack speed
    evasion = db.Column(db.Float, default=0)
    
    # Additional effects stored as JSON
    additional_effects = db.Column(db.JSON)
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type,
            'rarity': self.rarity,
            'stats': {
                'hp': self.hp,
                'atk': self.atk,
                'def': self.def_,
                'crit': self.crit,
                'atk_spd': self.atk_spd,
                'evasion': self.evasion
            },
            'additional_effects': self.additional_effects
        }
    
    @staticmethod
    def from_dict(data):
        equipment = Equipment(
            name=data['name'],
            type=data['type'],
            rarity=data.get('rarity'),
            hp=data['stats'].get('hp', 0),
            atk=data['stats'].get('atk', 0),
            def_=data['stats'].get('def', 0),
            crit=data['stats'].get('crit', 0),
            atk_spd=data['stats'].get('atk_spd', 0),
            evasion=data['stats'].get('evasion', 0),
            additional_effects=data.get('additional_effects', {})
        )
        return equipment 