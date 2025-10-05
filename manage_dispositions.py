#!/usr/bin/env python3
"""
Helper script to manage TFOPWG dispositions.
"""

import pandas as pd
import os

def show_disposition_codes():
    """Display available TFOPWG disposition codes."""
    print("TFOPWG Disposition Codes:")
    print("=" * 50)
    print("POSITIVE (Exoplanet):")
    print("  CP - Confirmed Planet")
    print("  KP - Known Planet")
    print("  PC - Planet Candidate")
    print("  AP - Ambiguous Planet")
    print()
    print("NEGATIVE (Not Exoplanet):")
    print("  FP - False Positive")
    print("  EB - Eclipsing Binary")
    print("  IS - Instrumental")
    print("  V  - Stellar Variability")
    print("  NTP - Not Transit-like Planet")
    print()

def show_current_dispositions():
    """Show current dispositions in the CSV file."""
    if not os.path.exists("tfopwg_dispositions.csv"):
        print("❌ No disposition file found. Run preprocess_data.py first.")
        return
    
    df = pd.read_csv("tfopwg_dispositions.csv")
    print(f"Current Dispositions ({len(df)} TIC IDs):")
    print("=" * 50)
    
    # Count by disposition
    disposition_counts = df['TFOPWG_Disposition'].value_counts()
    for disp, count in disposition_counts.items():
        print(f"  {disp}: {count} TIC IDs")
    
    print()
    print("Sample entries:")
    print(df.head(10).to_string(index=False))

def update_disposition(tic_id, disposition, notes=""):
    """Update a specific TIC ID's disposition."""
    if not os.path.exists("tfopwg_dispositions.csv"):
        print("❌ No disposition file found. Run preprocess_data.py first.")
        return
    
    df = pd.read_csv("tfopwg_dispositions.csv")
    
    if tic_id not in df['TIC_ID'].values:
        print(f"❌ TIC ID {tic_id} not found in disposition file.")
        return
    
    # Update the disposition
    df.loc[df['TIC_ID'] == tic_id, 'TFOPWG_Disposition'] = disposition
    if notes:
        df.loc[df['TIC_ID'] == tic_id, 'Notes'] = notes
    
    # Save back to CSV
    df.to_csv("tfopwg_dispositions.csv", index=False)
    print(f"✅ Updated TIC {tic_id} disposition to {disposition}")

def main():
    print("TFOPWG Disposition Manager")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Show disposition codes")
        print("2. Show current dispositions")
        print("3. Update disposition")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            show_disposition_codes()
        elif choice == '2':
            show_current_dispositions()
        elif choice == '3':
            tic_id = input("Enter TIC ID: ").strip()
            disposition = input("Enter disposition (CP/KP/PC/AP/FP/EB/IS/V/NTP): ").strip().upper()
            notes = input("Enter notes (optional): ").strip()
            update_disposition(tic_id, disposition, notes)
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
