package main

import (
	"time"
	"github.com/gin-gonic/gin"
	"github.com/gin-contrib/cors"
)

func SetupRoutes() *gin.Engine{
	// Для работы с БД будет исользоваться GORM из-за удобства автоматически создавать таблицы и записывать туда данные создавая их из type struct
	r := gin.Default()
	// Политика CORS чтобы доверять грубо говоря ip, пока что localhost можно поменять позже, пока не понятно где находится front в будущем
	config := cors.Config {
		AllowOrigins: []string{"http://localhost:3000"},
		AllowMethods: []string{"POST", "GET", "OPTIONS"},
		AllowHeaders: []string{"Origin", "Content-type", "Authorization"},
		AllowCredentials: true,
		MaxAge: 12 * time.Hour,
	}
	r.Use(cors.New(config))

	// позже настрою Cookie 
	r.GET("api/hello", Hello)
	r.GET("api/sessions/:session_id/messages")


	r.POST("api/sessions", CreateSessionHandler)
	r.POST("api/chat", UserMessageHandler)
	r.POST("api/leads", ManagerStatmentHandler)
	return r
}