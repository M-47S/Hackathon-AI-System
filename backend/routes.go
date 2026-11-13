package main

import (
	"net/http"
	"time"
	"errors"
	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

//Проверяем есть ли Cookie для необходимых эндпоинтов
func ReadCookieMiddleWare() gin.HandlerFunc {
	return func(ctx *gin.Context){
		cookie_id, err := ctx.Cookie(CookieName)
		if (err != nil || cookie_id == ""){
			ctx.AbortWithStatusJSON(401, FillErrorForm("User Anauthorized (No Cookie)", 401, err))
			return 
		} 
		cookie_struct, err := GetOneFromDB[Cookies]("cookie_id = ?", cookie_id, &Cookies{})
		if err != nil {
			ctx.AbortWithStatusJSON(http.StatusInternalServerError, FillErrorForm("User Anauthorized (DB Error)", http.StatusInternalServerError, err))
			return 
		}

		if cookie_struct.ExpiresAt.Before(time.Now()) {
			err = errors.New("Session expired")
			ctx.AbortWithStatusJSON(401, FillErrorForm("User Anauthorized (DB Error)", 401, err))
			return 
		}
		ctx.Set(CookieName, cookie_id)
		ctx.Next()
	}
}

func SetupRoutes() *gin.Engine {
	gin.SetMode(gin.DebugMode)
	r := gin.Default()
	// Политика CORS чтобы доверять грубо говоря ip, пока что localhost можно поменять позже, пока не понятно где находится front в будущем
	config := cors.Config {
		AllowOrigins: []string{"http://localhost:3000"},
		AllowMethods: []string{"POST", "GET", "OPTIONS"},
		AllowHeaders: []string{"Origin", "Content-Type", "Authorization"},
		AllowCredentials: true,
		MaxAge: 12 * time.Hour,
	}
	r.Use(cors.New(config))
	

	api := r.Group("/api")
	{
		// Публичные маршруты
		api.GET("/health", HealthHandler)
		api.POST("/logout", LogoutCustomer)
		api.POST("/register", RegisterCustomer)
		api.POST("/login/manual", LoginCustomerManual)

		// Группа с проверка на наличие Cookie
		cookieGroup := api.Group("", ReadCookieMiddleWare())
		{
			cookieGroup.GET("/login/cookie", LoginCustomerCookie)
			cookieGroup.POST("/sessions", CreateSessionHandler)
			cookieGroup.GET("/sessions/:session_id/messages", ChatHistoryHandler)
			cookieGroup.POST("/chat", UserMessageHandler)
			cookieGroup.POST("/leads", ManagerStatmentHandler)
		}
	}
	return r
}