import os
import yaml
from pathlib import Path
import argparse

#Written with assistance from Claude Sonnet

def find_repository_root(start_path=None):
    """
    Find the repository root by looking for common indicators like .git, setup.py, etc.
    Falls back to the current working directory if not found.
    """
    if start_path is None:
        start_path = Path.cwd()
    else:
        start_path = Path(start_path)
    
    # Common indicators of a repository root
    root_indicators = ['.git', 'setup.py', 'pyproject.toml', 'requirements.txt', 'README.md', 'mlruns']
    
    current = start_path.resolve()
    
    # Walk up the directory tree
    while current != current.parent:  # Stop at filesystem root
        # Check if any root indicators exist in current directory
        if any((current / indicator).exists() for indicator in root_indicators):
            print(f"Repository root detected: {current}")
            return current
        current = current.parent
    
    # If no indicators found, use the starting directory
    print(f"No clear repository root found, using: {start_path}")
    return start_path

def update_artifact_uris(repo_root=None, mlruns_dir="mlruns", experiment_id="607544705844869811", dry_run=False):
    """
    Update artifact_uri in all meta.yaml files in the mlruns_manuscript directory
    
    Args:
        repo_root: Path to repository root (auto-detected if None)
        mlruns_dir: Name of the mlruns directory (default: mlruns_manuscript)
        experiment_id: Experiment ID subdirectory
        dry_run: If True, show what would be changed without making changes
    """
    
    # Determine repository root
    if repo_root is None:
        repo_root = find_repository_root()
    else:
        repo_root = Path(repo_root).resolve()
    
    # Construct the path to the experiment directory
    experiment_path = repo_root / mlruns_dir / experiment_id
    
    if not experiment_path.exists():
        print(f"ERROR: Experiment directory does not exist: {experiment_path}")
        return False
    
    print(f"Repository root: {repo_root}")
    print(f"Experiment directory: {experiment_path}")
    print(f"Dry run mode: {dry_run}")
    print("-" * 60)
    
    # Find all run directories (subdirectories containing meta.yaml)
    run_dirs = []
    for item in experiment_path.iterdir():
        if item.is_dir():
            meta_file = item / "meta.yaml"
            if meta_file.exists():
                run_dirs.append(item)
    
    print(f"Found {len(run_dirs)} run directories with meta.yaml files")
    
    if not run_dirs:
        print("No run directories found with meta.yaml files")
        return False
    
    updated_count = 0
    error_count = 0
    
    # Process each run directory
    for run_dir in run_dirs:
        run_id = run_dir.name
        meta_file = run_dir / "meta.yaml"
        
        try:
            # Read the current meta.yaml
            with open(meta_file, 'r') as f:
                meta_data = yaml.safe_load(f)
            
            # Construct the new artifact URI
            new_artifact_uri = f"file://{repo_root}/{mlruns_dir}/{experiment_id}/{run_id}/artifacts"
            
            # Get current artifact_uri for comparison
            current_uri = meta_data.get('artifact_uri', '')
            
            print(f"\nRun ID: {run_id}")
            print(f"  Current artifact_uri: '{current_uri}'")
            print(f"  New artifact_uri:     '{new_artifact_uri}'")
            
            # Update the artifact_uri
            meta_data['artifact_uri'] = new_artifact_uri
            
            if not dry_run:
                # Write back the modified meta.yaml
                with open(meta_file, 'w') as f:
                    yaml.safe_dump(meta_data, f, default_flow_style=False)
                print(f"  ✓ Updated successfully")
            else:
                print(f"  (Would update in non-dry-run mode)")
            
            updated_count += 1
            
        except Exception as e:
            print(f"\nRun ID: {run_id}")
            print(f"  ✗ ERROR: {e}")
            error_count += 1
    
    print("\n" + "=" * 60)
    print(f"SUMMARY:")
    print(f"  Runs processed: {len(run_dirs)}")
    print(f"  Successfully updated: {updated_count}")
    print(f"  Errors: {error_count}")
    
    if dry_run:
        print(f"\nThis was a dry run. To actually update the files, run with --no-dry-run")
    
    return error_count == 0

def verify_updates(repo_root=None, mlruns_dir="mlruns", experiment_id="607544705844869811"):
    """
    Verify that all artifact URIs have been updated correctly
    """
    if repo_root is None:
        repo_root = find_repository_root()
    else:
        repo_root = Path(repo_root).resolve()
    
    experiment_path = repo_root / mlruns_dir / experiment_id
    
    if not experiment_path.exists():
        print(f"ERROR: Experiment directory does not exist: {experiment_path}")
        return False
    
    print(f"\nVerifying updates in: {experiment_path}")
    print("-" * 40)
    
    # Check all meta.yaml files
    valid_count = 0
    invalid_count = 0
    
    for item in experiment_path.iterdir():
        if item.is_dir():
            meta_file = item / "meta.yaml"
            if meta_file.exists():
                try:
                    with open(meta_file, 'r') as f:
                        meta_data = yaml.safe_load(f)
                    
                    artifact_uri = meta_data.get('artifact_uri', '')
                    expected_uri = f"file://{repo_root}/{mlruns_dir}/{experiment_id}/{item.name}/artifacts"
                    
                    if artifact_uri == expected_uri:
                        valid_count += 1
                        print(f"✓ {item.name}: Correct")
                    else:
                        invalid_count += 1
                        print(f"✗ {item.name}: Expected '{expected_uri}', got '{artifact_uri}'")
                        
                except Exception as e:
                    invalid_count += 1
                    print(f"✗ {item.name}: Error reading meta.yaml - {e}")
    
    print(f"\nVerification Summary:")
    print(f"  Valid URIs: {valid_count}")
    print(f"  Invalid URIs: {invalid_count}")
    
    return invalid_count == 0

def main():
    parser = argparse.ArgumentParser(description="Update MLFlow artifact URIs in meta.yaml files")
    parser.add_argument("--repo-root", type=str, help="Path to repository root (auto-detected if not specified)")
    parser.add_argument("--mlruns-dir", type=str, default="mlruns", help="MLRuns directory name")
    parser.add_argument("--experiment-id", type=str, default="607544705844869811", help="Experiment ID")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Show what would be changed without making changes")
    parser.add_argument("--no-dry-run", action="store_true", help="Actually make the changes")
    parser.add_argument("--verify", action="store_true", help="Verify that updates were applied correctly")
    
    args = parser.parse_args()
    
    # Handle dry-run logic
    dry_run = args.dry_run and not args.no_dry_run
    
    if args.verify:
        print("Running verification...")
        success = verify_updates(args.repo_root, args.mlruns_dir, args.experiment_id)
        if success:
            print("✓ All artifact URIs are correctly updated!")
        else:
            print("✗ Some artifact URIs need attention")
    else:
        print("Updating artifact URIs...")
        success = update_artifact_uris(args.repo_root, args.mlruns_dir, args.experiment_id, dry_run)
        if success and not dry_run:
            print("✓ All updates completed successfully!")
            
            # Optionally run verification after updates
            print("\nRunning verification...")
            verify_updates(args.repo_root, args.mlruns_dir, args.experiment_id)

if __name__ == "__main__":
    main()