pipeline {
    agent any
    stages {
        stage('Build & Test Model'){
            steps {
                sh 'chmod 777 scripts/train_test_log.sh'
                echo "Executing Training Script"
                sh './scripts/train_test_log.sh'
                echo "Training & Logging Complete"
            }
        }
        stage('Productionize'){
            steps {
                sh 'chmod 777 scripts/Productionize.sh'
                echo "Getting Model for Production"
                sh './scripts/Productionize.sh'
                echo "Best Model Updated"
            }
        }
        stage('Test Server'){
            steps {
                sh 'chmod 777 scripts/triton_test.sh'
                echo "Testing Triton Inference Server"
                sh './scripts/triton_test.sh'
                echo "Triton Server is working"
            }
        }
    }
}