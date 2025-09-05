#!/usr/bin/env python3
"""
Demo script for AutoPatternChecker Streamlit App
Creates sample data and shows how to use the app.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_sample_data():
    """Create sample data for demonstration."""
    
    # Sample data for different types of patterns
    data = []
    
    # Electronics components
    electronics_components = [
        ("Electronics", "Components", "Resistor", "1kΩ 1/4W"),
        ("Electronics", "Components", "Resistor", "2.2kΩ 1/2W"),
        ("Electronics", "Components", "Resistor", "10kΩ 1/4W"),
        ("Electronics", "Components", "Resistor", "100Ω 1/4W"),
        ("Electronics", "Components", "Resistor", "4.7kΩ 1/2W"),
        ("Electronics", "Components", "Capacitor", "100µF 25V"),
        ("Electronics", "Components", "Capacitor", "220µF 50V"),
        ("Electronics", "Components", "Capacitor", "47µF 16V"),
        ("Electronics", "Components", "Capacitor", "1000µF 35V"),
        ("Electronics", "Components", "Capacitor", "470µF 25V"),
        ("Electronics", "Components", "Inductor", "10mH 1A"),
        ("Electronics", "Components", "Inductor", "22mH 2A"),
        ("Electronics", "Components", "Inductor", "100µH 500mA"),
    ]
    
    # Tools and accessories
    tools_accessories = [
        ("Tools", "Accessories", "Connector", "SMB Mini Jack Right Angle"),
        ("Tools", "Accessories", "Connector", "SC Plug"),
        ("Tools", "Accessories", "Connector", "SMB Mini Jack Straight"),
        ("Tools", "Accessories", "Connector", "SC Plug Right Angle"),
        ("Tools", "Accessories", "Connector", "SMB Mini Jack Left Angle"),
        ("Tools", "Accessories", "Connector", "SC Plug Left Angle"),
        ("Tools", "Accessories", "Cable", "USB-C to USB-A 1m"),
        ("Tools", "Accessories", "Cable", "HDMI 2.1 2m"),
        ("Tools", "Accessories", "Cable", "Ethernet Cat6 5m"),
        ("Tools", "Accessories", "Cable", "Audio 3.5mm 2m"),
    ]
    
    # Mechanical parts
    mechanical_parts = [
        ("Mechanical", "Hardware", "Screw", "M3x10mm Phillips"),
        ("Mechanical", "Hardware", "Screw", "M4x15mm Phillips"),
        ("Mechanical", "Hardware", "Screw", "M3x8mm Hex"),
        ("Mechanical", "Hardware", "Screw", "M4x12mm Hex"),
        ("Mechanical", "Hardware", "Nut", "M3 Hex Nut"),
        ("Mechanical", "Hardware", "Nut", "M4 Hex Nut"),
        ("Mechanical", "Hardware", "Washer", "M3 Flat Washer"),
        ("Mechanical", "Hardware", "Washer", "M4 Flat Washer"),
        ("Mechanical", "Hardware", "Bolt", "M6x20mm Hex Bolt"),
        ("Mechanical", "Hardware", "Bolt", "M8x25mm Hex Bolt"),
    ]
    
    # Software licenses
    software_licenses = [
        ("Software", "Licenses", "Operating System", "Windows 11 Pro"),
        ("Software", "Licenses", "Operating System", "Ubuntu 22.04 LTS"),
        ("Software", "Licenses", "Operating System", "macOS Monterey"),
        ("Software", "Licenses", "Development", "Visual Studio Code"),
        ("Software", "Licenses", "Development", "PyCharm Professional"),
        ("Software", "Licenses", "Development", "IntelliJ IDEA"),
        ("Software", "Licenses", "Design", "Adobe Creative Suite"),
        ("Software", "Licenses", "Design", "Figma Professional"),
        ("Software", "Licenses", "Design", "Sketch"),
        ("Software", "Licenses", "Productivity", "Microsoft Office 365"),
    ]
    
    # Combine all data
    all_data = electronics_components + tools_accessories + mechanical_parts + software_licenses
    
    # Create DataFrame
    df = pd.DataFrame(all_data, columns=['key_part1', 'key_part2', 'key_part3', 'value'])
    
    # Add some variations to make it more interesting
    variations = []
    for _, row in df.iterrows():
        # Add original
        variations.append(row.to_dict())
        
        # Add some variations (with slight modifications)
        if np.random.random() < 0.3:  # 30% chance of variation
            value = row['value']
            if ' ' in value:
                # Add extra spaces
                varied_value = value.replace(' ', '  ')
                variations.append({
                    'key_part1': row['key_part1'],
                    'key_part2': row['key_part2'],
                    'key_part3': row['key_part3'],
                    'value': varied_value
                })
            elif value.isupper():
                # Add lowercase version
                variations.append({
                    'key_part1': row['key_part1'],
                    'key_part2': row['key_part2'],
                    'key_part3': row['key_part3'],
                    'value': value.lower()
                })
    
    # Create final DataFrame
    final_df = pd.DataFrame(variations)
    
    # Shuffle the data
    final_df = final_df.sample(frac=1).reset_index(drop=True)
    
    return final_df

def main():
    """Main demo function."""
    print("🚀 AutoPatternChecker Streamlit Demo")
    print("=" * 50)
    
    # Create sample data
    print("📊 Creating sample data...")
    sample_data = create_sample_data()
    
    # Save sample data
    sample_file = "sample_data_demo.csv"
    sample_data.to_csv(sample_file, index=False)
    
    print(f"✅ Sample data created: {sample_file}")
    print(f"   Shape: {sample_data.shape}")
    print(f"   Columns: {list(sample_data.columns)}")
    
    # Show sample data
    print("\n📋 Sample data preview:")
    print(sample_data.head(10).to_string(index=False))
    
    # Show data statistics
    print(f"\n📈 Data statistics:")
    print(f"   Total rows: {len(sample_data)}")
    print(f"   Unique keys: {len(sample_data.groupby(['key_part1', 'key_part2', 'key_part3']))}")
    print(f"   Value length range: {sample_data['value'].str.len().min()}-{sample_data['value'].str.len().max()}")
    
    # Show unique key combinations
    print(f"\n🔑 Unique key combinations:")
    unique_keys = sample_data.groupby(['key_part1', 'key_part2', 'key_part3']).size().reset_index(name='count')
    print(unique_keys.to_string(index=False))
    
    print(f"\n🎯 Next steps:")
    print(f"1. Run the Streamlit app: streamlit run streamlit_app.py")
    print(f"2. Upload the sample data: {sample_file}")
    print(f"3. Explore the patterns and train models")
    print(f"4. Test validation with sample values")
    
    print(f"\n🎉 Demo data ready! Start exploring with Streamlit!")

if __name__ == "__main__":
    main()