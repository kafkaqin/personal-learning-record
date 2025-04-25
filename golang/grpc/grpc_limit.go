package grpc

import (
	"context"

	gotimerate "golang.org/x/time/rate"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const (
	// DefaultQPS is determined by empirically reviewing known consumers of the API.
	// It's at least unlikely that there is a legitimate need to query podresources
	// more than 100 times per second, the other subsystems are not guaranteed to react
	// so fast in the first place.
	DefaultQPS = 100
	// DefaultBurstTokens is determined by empirically reviewing known consumers of the API.
	// See the documentation of DefaultQPS, same caveats apply.
	DefaultBurstTokens = 10
)

var (
	ErrorLimitExceeded = status.Error(codes.ResourceExhausted, "rejected by rate limit")
)

// Limiter defines the interface to perform request rate limiting,
// based on the interface exposed by https://pkg.go.dev/golang.org/x/time/rate#Limiter
type Limiter interface {
	// Allow reports whether an event may happen now.
	Allow() bool
}

// LimiterUnaryServerInterceptor returns a new unary server interceptors that performs request rate limiting.
func LimiterUnaryServerInterceptor(limiter Limiter) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		if !limiter.Allow() {
			return nil, ErrorLimitExceeded
		}
		return handler(ctx, req)
	}
}

func WithRateLimiter(qps, burstTokens int32) grpc.ServerOption {
	qpsVal := gotimerate.Limit(qps)
	burstVal := int(burstTokens)
	klog.InfoS("Setting rate limiting for podresources endpoint", "qps", qpsVal, "burstTokens", burstVal)
	return grpc.UnaryInterceptor(LimiterUnaryServerInterceptor(gotimerate.NewLimiter(qpsVal, burstVal)))
}
