pipeline {
    agent any
    stages {
        stage('Build Model'){
            steps {
                sh 'chmod +x scripts/build.sh'
                sh './scripts/build.sh'
            }
        }
    }
}