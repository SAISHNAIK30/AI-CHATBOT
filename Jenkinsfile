pipeline {
    agent any

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
                sh 'python kb.py'
            }
        }
    }

    post {
        always {
            echo "Pipeline finished!"
        }
    }
}
