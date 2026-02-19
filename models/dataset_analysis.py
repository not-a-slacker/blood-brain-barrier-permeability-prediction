# This file performs basic analysis of datasets used in our task - B3DB and BBBP.

import pandas as pd
# to ensure molecular descriptor table is not truncated in report
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, AllChem



class BBBDatasetAnalyzer:
    """Analyzer for Blood-Brain Barrier Permeability Dataset"""
    
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.df = None
        self.descriptors = None
        
    def load_data(self, file_name, file_type='csv'):
        """Load the dataset"""
        file_path = self.data_path / file_name
        
        if file_type == 'csv':
            self.df = pd.read_csv(file_path)
        elif file_type == 'tsv':
            self.df = pd.read_csv(file_path, sep='\t')

        # Create a figures subdirectory specific to this dataset (by file stem)
        self.current_dataset = Path(file_name).stem
        self.fig_dir = self.data_path.parent / 'figures' / self.current_dataset
        self.fig_dir.mkdir(parents=True, exist_ok=True)

        print(f"Dataset loaded: {file_path}")
        print(f"Shape: {self.df.shape}")
        return self.df
    
    def basic_stats(self):
        """Display basic statistics about the dataset"""
        
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Number of samples: {len(self.df)}")
        print(f"Number of features: {len(self.df.columns)}")
        
        print("\nColumn Names:")
        print(self.df.columns.tolist())
        
        print("\nFirst few rows:")
        print(self.df.head())
        
        print("\nData Types:")
        print(self.df.dtypes)
        
        print("\nMissing Values:")
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("No missing values!")
        
        print("\nDuplicate rows:")
        duplicates = self.df.duplicated().sum()
        print(f"Number of duplicate rows: {duplicates}")
        
    def analyze_class_distribution(self, target_col):
        """Analyze the distribution of BBB+/BBB- classes"""
        print("CLASS DISTRIBUTION ANALYSIS")
        
        # Count distribution
        class_counts = self.df[target_col].value_counts()
        print(f"\nClass Distribution:")
        print(class_counts)
        
        print(f"\nClass Proportions:")
        print(self.df[target_col].value_counts(normalize=True))
        
        
        imbalance_ratio = class_counts.max() / class_counts.min()
        print(f"\nImbalance Ratio: {imbalance_ratio:.2f}")
        
        # Visualize distribution
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Bar plot
        class_counts.plot(kind='bar', ax=axes[0], color=['red', 'green'])
        axes[0].set_title('Class Distribution (Bar)')
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Count')
        axes[0].tick_params(axis='x', rotation=0)
        
        # Pie chart
        class_counts.plot(kind='pie', ax=axes[1], autopct='%1.1f%%', 
                         colors=['green', 'red'])
        axes[1].set_title('Class Distribution (Pie)')
        axes[1].set_ylabel('')
        
        plt.tight_layout()
        out_path = self.fig_dir / 'class_distribution.png'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved: {out_path}")
        plt.show()
    
    def analyze_smiles(self, smiles_col='SMILES'):
        """Analyze SMILES strings"""
        print("SMILES ANALYSIS")
        
        
        # Basic SMILES statistics
        smiles_lengths = self.df[smiles_col].astype(str).str.len()
        
        print(f"\nSMILES Length Statistics:")
        print(f"Mean: {smiles_lengths.mean():.2f}")
        print(f"Median: {smiles_lengths.median():.2f}")
        print(f"Min: {smiles_lengths.min()}")
        print(f"Max: {smiles_lengths.max()}")
        print(f"Std: {smiles_lengths.std():.2f}")
        
        # Plot SMILES length distribution
        plt.figure(figsize=(10, 5))
        plt.hist(smiles_lengths, bins=50, color='blue', edgecolor='black', alpha=0.7)
        plt.xlabel('SMILES Length')
        plt.ylabel('Frequency')
        plt.title('Distribution of SMILES String Lengths')
        plt.grid(axis='y', alpha=0.3)
        out_path = self.fig_dir / 'smiles_length_distribution.png'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved: {out_path}")
        plt.show()
    
    def calculate_molecular_descriptors(self, smiles_col='SMILES', target_col=None):
        """Calculate molecular descriptors using RDKit"""
        
        print("MOLECULAR DESCRIPTORS ANALYSIS")
        
        
        descriptors_list = []
        
        print("\nCalculating molecular descriptors...")
        for idx, smiles in enumerate(self.df[smiles_col]):
            mol = Chem.MolFromSmiles(str(smiles))
            
            if mol is not None:
                desc = {
                    'MolWt': Descriptors.MolWt(mol),
                    'LogP': Crippen.MolLogP(mol),
                    'NumHDonors': Lipinski.NumHDonors(mol),
                    'NumHAcceptors': Lipinski.NumHAcceptors(mol),
                    'TPSA': Descriptors.TPSA(mol),
                    'NumRotatableBonds': Lipinski.NumRotatableBonds(mol),
                    'NumAromaticRings': Lipinski.NumAromaticRings(mol),
                    'NumHeteroatoms': Lipinski.NumHeteroatoms(mol),
                    'NumRings': Lipinski.RingCount(mol),
                    'FractionCsp3': Lipinski.FractionCSP3(mol),
                }
            else:
                desc = {key: np.nan for key in ['MolWt', 'LogP', 'NumHDonors', 
                       'NumHAcceptors', 'TPSA', 'NumRotatableBonds', 
                       'NumAromaticRings', 'NumHeteroatoms', 'NumRings', 'FractionCsp3']}
            
            descriptors_list.append(desc)
        
        self.descriptors = pd.DataFrame(descriptors_list)
        
        # Add target column if specified
        if target_col and target_col in self.df.columns:
            self.descriptors[target_col] = self.df[target_col].values
        
        print("\nMolecular Descriptors Summary:")
        print(self.descriptors.describe())
        
        return self.descriptors
    
    def visualize_descriptors(self, target_col=None):
        """Visualize molecular descriptors"""
        
        
        # Select numeric columns only (exclude target if present)
        numeric_cols = self.descriptors.select_dtypes(include=[np.number]).columns
        
        # 1. Distribution plots
        n_cols = 3
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, col in enumerate(numeric_cols):
            axes[idx].hist(self.descriptors[col].dropna(), bins=30, 
                          color='blue', edgecolor='black', alpha=0.7)
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'Distribution of {col}')
            axes[idx].grid(axis='y', alpha=0.3)
        
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        out_path = self.fig_dir / 'descriptor_distributions.png'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved: {out_path}")
        plt.show()
        
        # 2. Correlation heatmap
        plt.figure(figsize=(12, 10))
        corr_matrix = self.descriptors[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1)
        plt.title('Correlation Matrix of Molecular Descriptors')
        plt.tight_layout()
        out_path = self.fig_dir / 'descriptor_correlation.png'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {out_path}")
        plt.show()
        
        # 3. If target column exists, plot descriptors by class
        if target_col and target_col in self.descriptors.columns:
            print(f"\nAnalyzing descriptors by {target_col}...")
            
            top_descriptors = ['MolWt', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds']
            available_descriptors = [d for d in top_descriptors if d in self.descriptors.columns]
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for idx, descriptor in enumerate(available_descriptors):
                if idx < len(axes):
                    self.descriptors.boxplot(column=descriptor, by=target_col, ax=axes[idx])
                    axes[idx].set_title(f'{descriptor} by {target_col}')
                    axes[idx].set_xlabel(target_col)
                    axes[idx].set_ylabel(descriptor)
                    plt.sca(axes[idx])
                    plt.xticks(rotation=0)
            
            for idx in range(len(available_descriptors), len(axes)):
                axes[idx].axis('off')
            
            plt.suptitle('')  
            plt.tight_layout()
            out_path = self.fig_dir / 'descriptors_by_class.png'
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {out_path}")
            plt.show()
            
            # Statistical comparison between classes
            print(f"DESCRIPTOR STATISTICS BY {target_col.upper()}")
            
            for descriptor in available_descriptors:
                print(f"\n{descriptor}:")
                for class_val in self.descriptors[target_col].unique():
                    class_data = self.descriptors[self.descriptors[target_col] == class_val][descriptor]
                    print(f"  {class_val}: Mean={class_data.mean():.2f}, "
                          f"Median={class_data.median():.2f}, "
                          f"Std={class_data.std():.2f}")
    
    def generate_report(self, output_file='analysis_report.txt'):
        """Generate a text report of the analysis"""
        report_path = self.fig_dir / output_file
        
        with open(report_path, 'w') as f:
            f.write("Dataset Analysis Report\n")
            
            
            f.write("Dataset Information:\n")
            f.write(f"  Shape: {self.df.shape}\n")
            f.write(f"  Columns: {', '.join(self.df.columns)}\n\n")
            
            if self.descriptors is not None:
                f.write("Molecular Descriptors Summary:\n")
                desc = self.descriptors.describe()

                f.write(desc.to_string() + "\n\n")
            
            
        
        print(f"\nReport saved: {report_path}")


def main():
    """Main analysis function"""
    Path('../figures').mkdir(exist_ok=True)
    
    analyzer = BBBDatasetAnalyzer('../data')
    
    print("BLOOD-BRAIN BARRIER PERMEABILITY DATASET ANALYSIS")
    
    print("\n\n ANALYZING BBBP.csv \n")
    df_bbbp = analyzer.load_data('BBBP.csv', 'csv')
    analyzer.basic_stats()
    
    target_col = 'p_np' 
    smiles_col = 'SMILES'
    
    analyzer.analyze_class_distribution(target_col)
    
    analyzer.analyze_smiles(smiles_col)
    
    analyzer.calculate_molecular_descriptors(smiles_col, target_col)
    analyzer.visualize_descriptors(target_col)
    analyzer.generate_report('BBBP_analysis_report.txt')
    
    # Analyze B3DB dataset
    print("ANALYZING B3DB_classification.tsv")
    
    analyzer2 = BBBDatasetAnalyzer('../data')
    df_b3db = analyzer2.load_data('B3DB_classification.tsv', 'tsv')
    analyzer2.basic_stats()
    
    target_col_b3db = 'BBB+/BBB-'
    smiles_col_b3db = 'SMILES'
    
    analyzer2.analyze_class_distribution(target_col_b3db)
    
    analyzer2.analyze_smiles(smiles_col_b3db)
    
    analyzer2.calculate_molecular_descriptors(smiles_col_b3db, target_col_b3db)
    analyzer2.visualize_descriptors(target_col_b3db)
    analyzer2.generate_report('B3DB_analysis_report.txt')
    
if __name__ == "__main__":
    main()
