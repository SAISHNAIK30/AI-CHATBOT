pipeline {
    agent any

    stages {
        stage('Checkout Latest Code') {
            steps {
                echo "Pulling latest code from GitHub..."
                checkout scm
            }
        }

        stage('Notify') {
            steps {
                echo "Code has been updated and pulled locally. No scripts are run."
            }
        }
    }

    post {
        always {
            echo "Pipeline finished!"
        }
    }
}
