"""
Normalize Thailand stock data to match US format
Converts column names and removes unnecessary columns
"""

import os
import pandas as pd
from datetime import datetime

def normalize_thailand_data(input_dir='thailand_dataset', output_dir='thailand_dataset_normalized'):
    """
    Normalize Thailand stock data to match US format
    
    Input format (Thailand):
        Date, Open, High, Low, Close, Volume, Dividends, Stock Splits
    
    Output format (US compatible):
        date, open_price, high_price, low_price, close_price, volume_detail
    """
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all CSV files
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    print(f"Normalizing {len(csv_files)} Thailand stock files...")
    print("="*60)
    
    success_count = 0
    failed_count = 0
    
    for csv_file in csv_files:
        try:
            input_path = os.path.join(input_dir, csv_file)
            output_path = os.path.join(output_dir, csv_file)
            
            # Read data
            df = pd.read_csv(input_path)
            
            # Check if already normalized
            if 'date' in df.columns and 'close_price' in df.columns:
                print(f"✓ {csv_file} - Already normalized, skipping")
                continue
            
            # Rename columns to match US format
            column_mapping = {
                'Date': 'date',
                'Open': 'open_price',
                'High': 'high_price',
                'Low': 'low_price',
                'Close': 'close_price',
                'Volume': 'volume_detail'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Remove unnecessary columns
            columns_to_keep = ['date', 'open_price', 'high_price', 'low_price', 'close_price', 'volume_detail']
            df = df[columns_to_keep]
            
            # Convert date format (remove timezone info for consistency)
            df['date'] = pd.to_datetime(df['date'])
            if hasattr(df['date'].dtype, 'tz') and df['date'].dtype.tz is not None:
                df['date'] = df['date'].dt.tz_localize(None)
            
            # Sort by date ascending
            df = df.sort_values('date', ascending=True)
            
            # Save normalized data
            df.to_csv(output_path, index=False)
            
            print(f"✓ {csv_file} - Normalized {len(df)} rows")
            success_count += 1
            
        except Exception as e:
            print(f"✗ {csv_file} - Error: {str(e)}")
            failed_count += 1
    
    print("\n" + "="*60)
    print(f"Normalization complete!")
    print(f"Success: {success_count} files")
    print(f"Failed: {failed_count} files")
    print(f"Output directory: {output_dir}")
    
    return success_count, failed_count


def copy_selected_stocks(selected_file='classification/thailand_9_stocks.txt', 
                        source_dir='thailand_dataset_normalized',
                        output_dir='thailand_dataset_selected'):
    """
    Copy only the selected 9 stocks to a separate directory for training
    """
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read selected stocks
    with open(selected_file, 'r') as f:
        selected_stocks = [line.strip() for line in f if line.strip()]
    
    print(f"\nCopying {len(selected_stocks)} selected stocks to {output_dir}...")
    print("="*60)
    
    copied_count = 0
    for stock in selected_stocks:
        source_path = os.path.join(source_dir, f"{stock}.csv")
        output_path = os.path.join(output_dir, f"{stock}.csv")
        
        if os.path.exists(source_path):
            df = pd.read_csv(source_path)
            df.to_csv(output_path, index=False)
            print(f"✓ Copied {stock}.csv ({len(df)} rows)")
            copied_count += 1
        else:
            print(f"✗ {stock}.csv not found in {source_dir}")
    
    print("\n" + "="*60)
    print(f"Copied {copied_count}/{len(selected_stocks)} stocks successfully")
    
    return copied_count


if __name__ == "__main__":
    print("THAILAND DATA NORMALIZATION")
    print("="*60)
    
    # Step 1: Normalize all Thailand data
    success, failed = normalize_thailand_data()
    
    # Step 2: Copy selected 9 stocks
    if success > 0:
        copied = copy_selected_stocks()
        
        if copied > 0:
            print("\n✅ Thailand data is ready for training!")
            print("   Use directory: thailand_dataset_selected")
    else:
        print("\n❌ Normalization failed. Please check the input data.")
