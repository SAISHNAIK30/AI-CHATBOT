pipeline {
    agent any

    environment {
        PYTHON_HOME = 'C:\\Users\\SaishNaik\\AppData\\Local\\Programs\\Python\\Python313'
        PATH = "${env.PYTHON_HOME};${env.PYTHON_HOME}\\Scripts;${env.PATH}"
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Build') {
            steps {
                echo "Installing dependencies..."
                echo "No requirements.txt found"
            }
        }

        stage('Run') {
            steps {
                echo "Running Python script..."
                bat '"C:\\Users\\SaishNaik\\AppData\\Local\\Programs\\Python\\Python313\\python.exe" kb.py'
            }
        }
    }

    post {
        always {
            echo "Pipeline finished!"
        }
    }
}
