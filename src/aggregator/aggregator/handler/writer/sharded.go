// Copyright (c) 2020 Uber Technologies, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package writer

import (
	"errors"
	"fmt"
	"sync"

	"github.com/m3db/m3/src/aggregator/sharding"
	"github.com/m3db/m3/src/metrics/metric/aggregated"
	xerrors "github.com/m3db/m3/src/x/errors"
	"github.com/m3db/m3/src/x/instrument"
)

var (
	errShardedWriterNoWriters = errors.New("no backing writers provided")
	errShardedWriterClosed    = errors.New("sharded writer closed")
)

type shardedWriter struct {
	sync.RWMutex
	closed bool

	numShards int
	writers   []Writer
	locks     []sync.Mutex
	shardFn   sharding.AggregatedShardFn
}

var _ Writer = &shardedWriter{}

// NewShardedWriter shards writes to the provided writers with the given sharding fn.
func NewShardedWriter(
	writers []Writer,
	shardFn sharding.AggregatedShardFn,
	iOpts instrument.Options,
) (Writer, error) {
	if len(writers) == 0 {
		return nil, errShardedWriterNoWriters
	}

	return &shardedWriter{
		numShards: len(writers),
		writers:   writers,
		locks:     make([]sync.Mutex, len(writers)),
		shardFn:   shardFn,
	}, nil
}

func (w *shardedWriter) Write(mp aggregated.ChunkedMetricWithStoragePolicy) error {
	w.RLock()
	if w.closed {
		w.RUnlock()
		return errShardedWriterClosed
	}

	shardID := w.shardFn(mp.ChunkedID, w.numShards)
	w.locks[shardID].Lock()
	writerErr := w.writers[shardID].Write(mp)
	w.locks[shardID].Unlock()
	w.RUnlock()

	return writerErr
}

func (w *shardedWriter) Flush() error {
	w.RLock()
	defer w.RUnlock()

	if w.closed {
		return errShardedWriterClosed
	}

	var multiErr xerrors.MultiError
	for i := 0; i < w.numShards; i++ {
		w.locks[i].Lock()
		multiErr = multiErr.Add(w.writers[i].Flush())
		w.locks[i].Unlock()
	}

	if multiErr.Empty() {
		return nil
	}

	return fmt.Errorf("failed to flush sharded writer: %v", multiErr.FinalError())
}

func (w *shardedWriter) Close() error {
	w.Lock()
	defer w.Unlock()

	if w.closed {
		return errShardedWriterClosed
	}
	w.closed = true

	var multiErr xerrors.MultiError
	for i := 0; i < w.numShards; i++ {
		w.locks[i].Lock()
		multiErr = multiErr.Add(w.writers[i].Close())
		w.locks[i].Unlock()
	}

	if multiErr.Empty() {
		return nil
	}

	return fmt.Errorf("failed to close sharded writer: %v", multiErr.FinalError())
}
