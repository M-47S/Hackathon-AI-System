FROM golang:1.26-alpine AS builder

# Настраиваем Go-прокси, отключаем sumdb и HTTP/2 для избежания таймаутов
ENV GOPROXY=https://goproxy.cn,direct \
    GOSUMDB=off \
    GODEBUG=http2client=0 \
    CGO_ENABLED=0 \
    GOOS=linux

WORKDIR /mts

COPY go.mod go.sum ./
RUN go mod download && go mod verify

COPY . .
RUN go build -o server .

FROM alpine:latest

WORKDIR /mts
COPY --from=builder /mts/server .
COPY .env .

EXPOSE 8080
CMD ["./server"]