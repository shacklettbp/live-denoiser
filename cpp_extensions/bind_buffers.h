#ifndef BIND_BUFFERS_H_INCLUDED
#define BIND_BUFFERS_H_INCLUDED


std::pair<void *, void *> getDevicePtr(void *handle, uint64_t bytes);

#endif
