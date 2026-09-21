import os
import glob
import csv
import xml.etree.ElementTree as ET

# Dictionary to map German DFL positions to EA FC 26 standard positions
POSITION_MAPPING = {
    'TW': 'GK',
    'LV': 'LB',
    'IVL': 'CB',
    'IVZ': 'CB',
    'IVR': 'CB',
    'RV': 'RB',
    'DML': 'CDM',
    'DMZ': 'CDM',
    'DMR': 'CDM',
    'DLM': 'CDM',  # Added Defensive Left Mid
    'DRM': 'CDM',  # Added Defensive Right Mid
    'LM': 'LM',
    'HL': 'CM',
    'MZ': 'CM',
    'HR': 'CM',
    'RM': 'RM',
    'OLM': 'CAM',
    'ZO': 'CAM',
    'ORM': 'CAM',
    'LA': 'LW',
    'STL': 'ST',
    'HST': 'ST',  # Hängende Spitze (Second Striker) mapped to ST
    'STZ': 'ST',
    'STR': 'ST',
    'RA': 'RW'
}

def parse_and_write_csv(xml_files, csv_path):
    records = set() # To avoid duplicates
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Find all Team nodes
            for team in root.iter('Team'):
                team_id = team.get('TeamId')
                team_name = team.get('TeamName')
                
                # Find all Player nodes within this Team
                for player in team.iter('Player'):
                    player_id = player.get('PersonId')
                    real_name = player.get('Shortname')
                    shirt_number = player.get('ShirtNumber')
                    original_position = player.get('PlayingPosition')
                    
                    # Map the position; use original if not found in dictionary
                    player_position = POSITION_MAPPING.get(original_position, original_position)
                    
                    # Update logic to look for the new 'GK' string instead of 'TW'
                    if player_position == 'GK':
                        is_goalkeeper = 1
                    else:
                        is_goalkeeper = 0
                        
                    if shirt_number is not None:
                        shirt_number = int(shirt_number)
                    else:
                        shirt_number = -1 # Indicate missing shirt number clearly
                        
                    records.add((player_id, real_name, shirt_number, is_goalkeeper, original_position, player_position, team_id, team_name))
        except Exception as e:
            print(f"Error parsing {xml_file}: {e}")

    # Write to CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['player_id', 'real_name', 'shirt_number', 'is_goalkeeper', 'original_position', 'player_position', 'team_id', 'team_name'])
        writer.writerows(list(records))
    
    print(f"Computed {len(records)} unique player-team-shirt records.")
    print(f"Database created successfully at {csv_path}")

if __name__ == '__main__':
    base_dir = r"data/raw"
    csv_path = r"data/processed/players_database.csv"
    
    # Find all the DFL match information XMLs
    xml_files = glob.glob(os.path.join(base_dir, "*matchinformation*.xml"))
    print(f"Found {len(xml_files)} match information XML files.")
    
    parse_and_write_csv(xml_files, csv_path)