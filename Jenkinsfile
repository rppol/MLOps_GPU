pipeline {
    agent any
    stages {
        stage('Train') {
            steps {
                chmod +x scripts/train.sh
                sh './scripts/train.sh'
            }
        }
    }
}