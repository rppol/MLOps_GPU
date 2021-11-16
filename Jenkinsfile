pipeline {
    agent any
    stages {
        stage('Build & Test Model'){
            steps {
                sh 'chmod 777 scripts/train_test_log.sh'
                sh './scripts/train_test_log.sh'
            }
        }
        stage('Get Best Model'){
            steps {
                sh 'chmod 777 scripts/get_best_model.sh'
                sh './scripts/get_best_model.sh'
            }
        }
        stage('Deploy'){
            steps {
                
            }
        }
    }
}