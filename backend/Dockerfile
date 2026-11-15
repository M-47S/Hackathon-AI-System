FROM golang:1.26-alpine AS builder

WORKDIR /mts


COPY go.mod go.sum ./
RUN go mod download


COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o server .


FROM alpine:latest

WORKDIR /mts
COPY --from=builder /mts/server .
COPY .env .

EXPOSE 8080
CMD ["./server"]
