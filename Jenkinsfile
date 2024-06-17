pipeline {
    agent any
    environment {
      OPENAI_API_KEY = credentials('openai-api-key')
      YOUTUBE_DEVELOPER_KEY = credentials('youtube-developer-key')
      DOCKER_IMAGE = "chatbot"
      DOCKER_CONTAINER_NAME = "chatbot"
    }
    options {
        skipStagesAfterUnstable()
    }
    stages {
        stage('Clone repository') {
            steps {
                script {
                    checkout scm
                }
            }
        }

        stage('Build') {
            steps {
                script {
                    app = docker.build("${DOCKER_IMAGE}")
                }
            }
        }

        
        stage('run the containers') {
            steps {
                script {
                    // Stop any running container with the same name
                    sh "docker stop ${DOCKER_CONTAINER_NAME} || true"
                    sh "docker rm ${DOCKER_CONTAINER_NAME} || true"
                    
                    // Run the new container
                    sh('docker run -d -e OPENAI_API_KEY=$OPENAI_API_KEY -e DEVELOPER_KEY=$YOUTUBE_DEVELOPER_KEY --name $DOCKER_CONTAINER_NAME -p 8000:8000 $DOCKER_IMAGE')
                }
            }
        }

        stage('Test') {
            steps {
                sh 'docker ps'
            } 
        }
    }
    post {
        always {
            cleanWs()
        }
    }
}
