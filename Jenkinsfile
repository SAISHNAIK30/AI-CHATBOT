pipeline {
    agent any

    environment {
        // Replace with your actual Python path
        PYTHON_HOME = 'C:\Users\SaishNaik\AppData\Local\Programs\Python\Python313'''
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
                bat 'python kb.py'
            }
        }
    }

    post {
        always {
            echo "Pipeline finished!"
        }
    }
}
