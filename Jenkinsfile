pipeline {
    agent any
    stages {
        stage('Build & Test Model'){
            steps {
                sh 'chmod 777 scripts/train_test_log.sh'
                sh './scripts/train_test_log.sh'
            }
        }
        stage('Productionize'){
            steps {
                sh 'chmod 777 scripts/Productionize.sh'
                sh './scripts/Productionize.sh'
            }
        }
        stage('Test Server'){
            steps {
                sh 'chmod 777 scripts/triton_test.sh'
                sh './scripts/triton_test.sh'
            }
        }
    }
}