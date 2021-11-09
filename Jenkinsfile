pipeline {
    agent any
    stages {
        stage('Train'){
            steps {
                sh 'chmod +x /scripts/train.sh'
                sh './scripts/train.sh'
            }
        }
    }
}