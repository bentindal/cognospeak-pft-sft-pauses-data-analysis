"""File naming convention validation for transcript files."""

import re
from collections import defaultdict
from pathlib import Path


def validate_file_naming(data_dir: Path) -> dict[str, list[str]]:
    """
    Check for files that don't follow naming convention.
    
    Files should be named: Process-XXX__PFT.txt or Process-XXX__SFT.txt (double underscore)
    OR Process-XXX_PFT.txt or Process-XXX_SFT.txt (single underscore) - both are acceptable
    Files should be in directories: Process-rec-XXX/ or Process-test-XXX/
    
    Args:
        data_dir: Path to the data directory
        
    Returns:
        Dictionary with 'wrong_task_type' and 'missing_task_type' lists
    """
    naming_issues = defaultdict(list)
    
    # Check all files in data directory
    for file_path in data_dir.rglob('*'):
        if file_path.is_file() and file_path.name != 'speechpause_mata.csv':
            # Check if file has PFT or SFT in name
            has_task = bool(re.search(r'[PS]FT', file_path.name, re.IGNORECASE))
            
            if has_task:
                # Check for wrong task type (not PFT or SFT after underscore)
                # Check both single and double underscore patterns
                if re.search(r'__[^PS]', file_path.name, re.IGNORECASE) or \
                   re.search(r'(?<!_)_[^PS]FT', file_path.name, re.IGNORECASE):
                    naming_issues['wrong_task_type'].append(str(file_path))
            
            # Check for files without PFT or SFT
            elif file_path.suffix in ['.txt', '.wav']:
                naming_issues['missing_task_type'].append(str(file_path))
    
    return naming_issues


def print_validation_report(naming_issues: dict[str, list[str]]) -> None:
    """
    Print a formatted report of file naming issues.
    
    Args:
        naming_issues: Dictionary returned from validate_file_naming()
    """
    print("=" * 80)
    print("FILE NAMING CONVENTION CHECK")
    print("=" * 80)
    print(f"\nAccepted formats:")
    print(f"  - Process-XXX__PFT.txt or Process-XXX__SFT.txt (double underscore)")
    print(f"  - Process-XXX_PFT.txt or Process-XXX_SFT.txt (single underscore)")
    print(f"Files should be in directories: Process-rec-XXX/ or Process-test-XXX/\n")
    
    if naming_issues['wrong_task_type']:
        print(f"❌ FILES WITH WRONG TASK TYPE (should be PFT or SFT):")
        for issue in naming_issues['wrong_task_type']:
            print(f"   - {issue}")
        print()
    
    if naming_issues['missing_task_type']:
        print(f"⚠️  FILES WITHOUT PFT/SFT IN NAME:")
        for issue in naming_issues['missing_task_type']:
            print(f"   - {issue}")
        print()
    
    if not any(naming_issues.values()):
        print("✅ All files follow the naming convention!")
    else:
        print(f"\nTotal issues found: {sum(len(v) for v in naming_issues.values())} files")
    print("=" * 80)

