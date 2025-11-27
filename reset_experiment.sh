#!/bin/bash
# Reset Experiment Script
# Deletes all generated results and intermediate files from the experiment

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "  Reset Experiment - Delete Stored Results"
echo "=========================================="
echo ""

# List of files/directories to delete
FILES_TO_DELETE=(
    "output/"
    "pause_summary.csv"
    "combined_transcripts.parquet"
    "__pycache__/"
)

# Count files that exist
EXISTING_COUNT=0
for item in "${FILES_TO_DELETE[@]}"; do
    if [ -e "$item" ]; then
        EXISTING_COUNT=$((EXISTING_COUNT + 1))
    fi
done

if [ $EXISTING_COUNT -eq 0 ]; then
    echo -e "${GREEN}✓ No stored results found. Nothing to delete.${NC}"
    exit 0
fi

# Show what will be deleted
echo "The following files/directories will be deleted:"
for item in "${FILES_TO_DELETE[@]}"; do
    if [ -e "$item" ]; then
        if [ -d "$item" ]; then
            echo -e "  ${RED}🗑  $item${NC} (directory)"
        else
            echo -e "  ${RED}🗑  $item${NC}"
        fi
    fi
done
echo ""

# Confirmation prompt
read -p "Are you sure you want to delete these files? (yes/no): " -r
echo ""

if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo -e "${YELLOW}Cancelled. No files were deleted.${NC}"
    exit 0
fi

# Delete files
DELETED_COUNT=0
for item in "${FILES_TO_DELETE[@]}"; do
    if [ -e "$item" ]; then
        if [ -d "$item" ]; then
            rm -rf "$item"
            echo -e "${GREEN}✓ Deleted directory: $item${NC}"
        else
            rm -f "$item"
            echo -e "${GREEN}✓ Deleted file: $item${NC}"
        fi
        DELETED_COUNT=$((DELETED_COUNT + 1))
    fi
done

echo ""
echo -e "${GREEN}=========================================="
echo "  Reset Complete!"
echo "==========================================${NC}"
echo -e "Deleted ${DELETED_COUNT} item(s)."
echo ""
echo "You can now run the experiment again from scratch."

