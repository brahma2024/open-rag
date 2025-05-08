#!/bin/bash
set -e  # Exit immediately if any command fails

echo "🚀 Starting deployment to Development Environment..."

DEPLOY_PATH=~/deployments/develop
REPO_PATH=~/fintrade

mkdir -p $DEPLOY_PATH

echo "🔄 Pulling latest changes from develop branch..."
cd $REPO_PATH
git checkout develop
git pull origin develop

echo "📁 Syncing files to Development environment..."
rsync -av --exclude '.git' --exclude '__pycache__' --exclude 'data/logs' ./ $DEPLOY_PATH/

echo "📦 Setting up virtual environment..."
cd $DEPLOY_PATH

# ✅ Check if venv already exists
if [ ! -d "venv" ]; then
    echo "🆕 Creating a new virtual environment..."
    python -m venv venv
else
    echo "✅ Virtual environment already exists. Skipping recreation."
fi

# Activate virtual environment
source venv/bin/activate

# ✅ Install dependencies only if requirements have changed
if ! cmp -s $REPO_PATH/requirements.txt $DEPLOY_PATH/requirements.txt; then
    echo "📦 Installing updated dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    cp $REPO_PATH/requirements.txt $DEPLOY_PATH/requirements.txt  # Cache the installed version
else
    echo "✅ Dependencies are already up-to-date. Skipping installation."
fi

echo "🛠️ Running database migrations (if applicable)..."
if [ -f "scripts/migrate.sh" ]; then
    ./scripts/migrate.sh
else
    echo "⚠️ No migration script found, skipping..."
fi

echo "🚀 Restarting application (if applicable)..."
if pgrep -f "my_app.py" > /dev/null; then
    echo "🔄 Restarting running process..."
    pkill -f "my_app.py"
fi
nohup python src/main.py &

echo "✅ Deployment to Development Environment Complete!"
