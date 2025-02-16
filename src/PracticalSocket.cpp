/*
 *   C++ sockets on Unix and Windows
 *   Copyright (C) 2002
 *
 *   This program is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation; either version 2 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program; if not, write to the Free Software
 *   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#include "PracticalSocket.h"
#include <cstring>

#ifdef _WIN32
#include <winsock2.h>         // For socket(), connect(), send(), and recv()
typedef int socklen_t;
typedef char raw_type;       // Type used for raw data on this platform
#pragma comment(lib, "Ws2_32.lib")
#else
#include <sys/types.h>       // For data types
#include <sys/socket.h>      // For socket(), connect(), send(), and recv()
#include <netdb.h>           // For gethostbyname()
#include <arpa/inet.h>       // For inet_addr()
#include <unistd.h>          // For close()
#include <netinet/in.h>      // For sockaddr_in
typedef void raw_type;       // Type used for raw data on this platform
#endif

#include <cerrno>             // For errno
#include <iostream>           // For cerr

using namespace std;

#ifdef _WIN32
static bool initialized = false;
#endif

// SocketException Code

SocketException::SocketException(const string& message, bool inclSysMsg) noexcept
    : userMessage(message) {
    if (inclSysMsg) {
        userMessage.append(": ");
        userMessage.append(strerror(errno));
    }
}

SocketException::~SocketException() noexcept = default;

const char* SocketException::what() const noexcept {
    return userMessage.c_str();
}

// Function to fill in address structure given an address and port
static void fillAddr(const string& address, unsigned short port, sockaddr_in& addr) {
    memset(&addr, 0, sizeof(addr));  // Zero out address structure
    addr.sin_family = AF_INET;       // Internet address

    hostent* host;  // Resolve name
    if ((host = gethostbyname(address.c_str())) == nullptr) {
        throw SocketException("Failed to resolve name (gethostbyname())");
    }
    addr.sin_addr.s_addr = *reinterpret_cast<unsigned long*>(host->h_addr_list[0]);
    addr.sin_port = htons(port);     // Assign port in network byte order
}

// Socket Code

Socket::Socket(int type, int protocol) {
#ifdef _WIN32
    if (!initialized) {
        WORD wVersionRequested = MAKEWORD(2, 0); // Request WinSock v2.0
        WSADATA wsaData;
        if (WSAStartup(wVersionRequested, &wsaData) != 0) { // Load WinSock DLL
            throw SocketException("Unable to load WinSock DLL");
        }
        initialized = true;
    }
#endif

    // Make a new socket
    if ((sockDesc = socket(PF_INET, type, protocol)) < 0) {
        throw SocketException("Socket creation failed (socket())", true);
    }
}

Socket::Socket(int sockDesc) : sockDesc(sockDesc) {}

Socket::~Socket() {
#ifdef _WIN32
    ::closesocket(sockDesc);
#else
    ::close(sockDesc);
#endif
    sockDesc = -1;
}

string Socket::getLocalAddress() const {
    sockaddr_in addr;
    socklen_t addr_len = sizeof(addr);

    if (getsockname(sockDesc, reinterpret_cast<sockaddr*>(&addr), &addr_len) < 0) {
        throw SocketException("Fetch of local address failed (getsockname())", true);
    }
    return inet_ntoa(addr.sin_addr);
}

unsigned short Socket::getLocalPort() const {
    sockaddr_in addr;
    socklen_t addr_len = sizeof(addr);

    if (getsockname(sockDesc, reinterpret_cast<sockaddr*>(&addr), &addr_len) < 0) {
        throw SocketException("Fetch of local port failed (getsockname())", true);
    }
    return ntohs(addr.sin_port);
}

void Socket::setLocalPort(unsigned short localPort) {
    sockaddr_in localAddr;
    memset(&localAddr, 0, sizeof(localAddr));
    localAddr.sin_family = AF_INET;
    localAddr.sin_addr.s_addr = htonl(INADDR_ANY);
    localAddr.sin_port = htons(localPort);

    if (bind(sockDesc, reinterpret_cast<sockaddr*>(&localAddr), sizeof(sockaddr_in)) < 0) {
        throw SocketException("Set of local port failed (bind())", true);
    }
}

void Socket::setLocalAddressAndPort(const string& localAddress, unsigned short localPort) {
    sockaddr_in localAddr;
    fillAddr(localAddress, localPort, localAddr);

    if (bind(sockDesc, reinterpret_cast<sockaddr*>(&localAddr), sizeof(sockaddr_in)) < 0) {
        throw SocketException("Set of local address and port failed (bind())", true);
    }
}

void Socket::cleanUp() {
#ifdef _WIN32
    if (WSACleanup() != 0) {
        throw SocketException("WSACleanup() failed");
    }
#endif
}

unsigned short Socket::resolveService(const string& service, const string& protocol) {
    struct servent* serv;
    if ((serv = getservbyname(service.c_str(), protocol.c_str())) == nullptr) {
        return static_cast<unsigned short>(stoi(service));  // Service is port number
    } else {
        return ntohs(serv->s_port);  // Found port (network byte order) by name
    }
}

// CommunicatingSocket Code

CommunicatingSocket::CommunicatingSocket(int type, int protocol) : Socket(type, protocol) {}

CommunicatingSocket::CommunicatingSocket(int newConnSD) : Socket(newConnSD) {}

void CommunicatingSocket::connect(const string& foreignAddress, unsigned short foreignPort) {
    sockaddr_in destAddr;
    fillAddr(foreignAddress, foreignPort, destAddr);

    if (::connect(sockDesc, reinterpret_cast<sockaddr*>(&destAddr), sizeof(destAddr)) < 0) {
        throw SocketException("Connect failed (connect())", true);
    }
}

void CommunicatingSocket::send(const void* buffer, int bufferLen) {
    if (::send(sockDesc, reinterpret_cast<const raw_type*>(buffer), bufferLen, 0) < 0) {
        throw SocketException("Send failed (send())", true);
    }
}

int CommunicatingSocket::recv(void* buffer, int bufferLen) {
    int rtn;
    if ((rtn = ::recv(sockDesc, reinterpret_cast<raw_type*>(buffer), bufferLen, 0)) < 0) {
        throw SocketException("Receive failed (recv())", true);
    }
    return rtn;
}

string CommunicatingSocket::getForeignAddress() const {
    sockaddr_in addr;
    socklen_t addr_len = sizeof(addr);

    if (getpeername(sockDesc, reinterpret_cast<sockaddr*>(&addr), &addr_len) < 0) {
        throw SocketException("Fetch of foreign address failed (getpeername())", true);
    }
    return inet_ntoa(addr.sin_addr);
}

unsigned short CommunicatingSocket::getForeignPort() const {
    sockaddr_in addr;
    socklen_t addr_len = sizeof(addr);

    if (getpeername(sockDesc, reinterpret_cast<sockaddr*>(&addr), &addr_len) < 0) {
        throw SocketException("Fetch of foreign port failed (getpeername())", true);
    }
    return ntohs(addr.sin_port);
}

// TCPSocket Code

TCPSocket::TCPSocket() : CommunicatingSocket(SOCK_STREAM, IPPROTO_TCP) {}

TCPSocket::TCPSocket(const string& foreignAddress, unsigned short foreignPort) : CommunicatingSocket(SOCK_STREAM, IPPROTO_TCP) {
    connect(foreignAddress, foreignPort);
}

TCPSocket::TCPSocket(int newConnSD) : CommunicatingSocket(newConnSD) {}

// TCPServerSocket Code

TCPServerSocket::TCPServerSocket(unsigned short localPort, int queueLen) : Socket(SOCK_STREAM, IPPROTO_TCP) {
    setLocalPort(localPort);
    setListen(queueLen);
}

TCPServerSocket::TCPServerSocket(const string& localAddress, unsigned short localPort, int queueLen) : Socket(SOCK_STREAM, IPPROTO_TCP) {
    setLocalAddressAndPort(localAddress, localPort);
    setListen(queueLen);
}

TCPSocket* TCPServerSocket::accept() {
    int newConnSD;
    if ((newConnSD = ::accept(sockDesc, nullptr, nullptr)) < 0) {
        throw SocketException("Accept failed (accept())", true);
    }
    return new TCPSocket(newConnSD);
}

void TCPServerSocket::setListen(int queueLen) {
    if (listen(sockDesc, queueLen) < 0) {
        throw SocketException("Set listening socket failed (listen())", true);
    }
}

// UDPSocket Code

UDPSocket::UDPSocket() : CommunicatingSocket(SOCK_DGRAM, IPPROTO_UDP) {
    setBroadcast();
}

UDPSocket::UDPSocket(unsigned short localPort) : CommunicatingSocket(SOCK_DGRAM, IPPROTO_UDP) {
    setLocalPort(localPort);
    setBroadcast();
}

UDPSocket::UDPSocket(const string& localAddress, unsigned short localPort) : CommunicatingSocket(SOCK_DGRAM, IPPROTO_UDP) {
    setLocalAddressAndPort(localAddress, localPort);
    setBroadcast();
}

void UDPSocket::setBroadcast() {
    int broadcastPermission = 1;
    setsockopt(sockDesc, SOL_SOCKET, SO_BROADCAST, reinterpret_cast<raw_type*>(&broadcastPermission), sizeof(broadcastPermission));
}

void UDPSocket::disconnect() {
    sockaddr_in nullAddr;
    memset(&nullAddr, 0, sizeof(nullAddr));
    nullAddr.sin_family = AF_UNSPEC;

    if (::connect(sockDesc, reinterpret_cast<sockaddr*>(&nullAddr), sizeof(nullAddr)) < 0) {
#ifdef _WIN32
        if (errno != WSAEAFNOSUPPORT) {
#else
        if (errno != EAFNOSUPPORT) {
#endif
            throw SocketException("Disconnect failed (connect())", true);
        }
    }
}

void UDPSocket::sendTo(const void* buffer, int bufferLen, const string& foreignAddress, unsigned short foreignPort) {
    sockaddr_in destAddr;
    fillAddr(foreignAddress, foreignPort, destAddr);

    if (sendto(sockDesc, reinterpret_cast<const raw_type*>(buffer), bufferLen, 0, reinterpret_cast<sockaddr*>(&destAddr), sizeof(destAddr)) != bufferLen) {
        throw SocketException("Send failed (sendto())", true);
    }
}

int UDPSocket::recvFrom(void* buffer, int bufferLen, string& sourceAddress, unsigned short& sourcePort) {
    sockaddr_in clntAddr;
    socklen_t addrLen = sizeof(clntAddr);
    int rtn;
    if ((rtn = recvfrom(sockDesc, reinterpret_cast<raw_type*>(buffer), bufferLen, 0, reinterpret_cast<sockaddr*>(&clntAddr), &addrLen)) < 0) {
        throw SocketException("Receive failed (recvfrom())", true);
    }
    sourceAddress = inet_ntoa(clntAddr.sin_addr);
    sourcePort = ntohs(clntAddr.sin_port);
    return rtn;
}

void UDPSocket::setMulticastTTL(unsigned char multicastTTL) {
    if (setsockopt(sockDesc, IPPROTO_IP, IP_MULTICAST_TTL, reinterpret_cast<raw_type*>(&multicastTTL), sizeof(multicastTTL)) < 0) {
        throw SocketException("Multicast TTL set failed (setsockopt())", true);
    }
}

void UDPSocket::joinGroup(const string& multicastGroup) {
    ip_mreq multicastRequest;
    multicastRequest.imr_multiaddr.s_addr = inet_addr(multicastGroup.c_str());
    multicastRequest.imr_interface.s_addr = htonl(INADDR_ANY);

    if (setsockopt(sockDesc, IPPROTO_IP, IP_ADD_MEMBERSHIP, reinterpret_cast<raw_type*>(&multicastRequest), sizeof(multicastRequest)) < 0) {
        throw SocketException("Multicast group join failed (setsockopt())", true);
    }
}

void UDPSocket::leaveGroup(const string& multicastGroup) {
    ip_mreq multicastRequest;
    multicastRequest.imr_multiaddr.s_addr = inet_addr(multicastGroup.c_str());
    multicastRequest.imr_interface.s_addr = htonl(INADDR_ANY);

    if (setsockopt(sockDesc, IPPROTO_IP, IP_DROP_MEMBERSHIP, reinterpret_cast<raw_type*>(&multicastRequest), sizeof(multicastRequest)) < 0) {
        throw SocketException("Multicast group leave failed (setsockopt())", true);
    }
}
