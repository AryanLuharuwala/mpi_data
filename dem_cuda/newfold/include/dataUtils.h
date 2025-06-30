#ifndef DATA_UTILS_H
#define DATA_UTILS_H

#include <mpi.h>
#include <vector>
#include <type_traits>
#include <string>
#include <utility>
#include <map>
#include <set>
#include <list>
#include <array>
#include <unordered_map>
#include <unordered_set>

// First, define the C++17 void_t for C++11 compatibility
template<typename... Ts>
struct make_void { typedef void type; };

template<typename... Ts>
using void_t = typename make_void<Ts...>::type;

// Type traits to detect container types
template <typename T, typename = void>
struct is_container : std::false_type {};

template <typename T>
struct is_container<T, 
    void_t<
        typename T::value_type,
        typename T::size_type,
        typename T::iterator,
        decltype(std::declval<T>().begin()),
        decltype(std::declval<T>().end()),
        decltype(std::declval<T>().size())
    >> : std::true_type {};

// Type trait for std::vector specifically
template <typename T>
struct is_vector : std::false_type {};

template <typename T, typename A>
struct is_vector<std::vector<T, A>> : std::true_type {};

// Type trait for std::string
template <typename T>
struct is_string : std::false_type {};

template <typename C, typename T, typename A>
struct is_string<std::basic_string<C, T, A>> : std::true_type {};

// Type trait for std::pair
template <typename T>
struct is_pair : std::false_type {};

template <typename T1, typename T2>
struct is_pair<std::pair<T1, T2>> : std::true_type {};

// Type trait for map-like types
template <typename T>
struct is_map_like : std::false_type {};

template <typename K, typename V, typename... Args>
struct is_map_like<std::map<K, V, Args...>> : std::true_type {};

template <typename K, typename V, typename... Args>
struct is_map_like<std::unordered_map<K, V, Args...>> : std::true_type {};

// Type trait for detecting custom serialization interface - MOVED HERE
template <typename T, typename = void>
struct has_serialize_method : std::false_type {};

template <typename T>
struct has_serialize_method<T, 
    void_t<decltype(std::declval<T>().serialize())>> : std::true_type {};

template <typename T, typename = void>
struct has_deserialize_method : std::false_type {};

template <typename T>
struct has_deserialize_method<T, 
    void_t<decltype(T::deserialize(std::declval<typename T::serialized_type>()))>> : std::true_type {};

// Forward declarations to avoid ambiguity
template <typename T>
typename std::enable_if<std::is_trivially_copyable<T>::value, void>::type
sendDataSimple(const std::vector<T>& data, int dest_rank, int tag = 0);

template <typename T>
void sendDataNested(const std::vector<std::vector<T>>& data, int dest_rank, int tag = 0);

template <typename T>
typename std::enable_if<std::is_trivially_copyable<T>::value, std::vector<T>>::type
receiveDataSimple(int source_rank, int tag = 0);

template <typename T>
std::vector<std::vector<T>> receiveDataNested(int source_rank, int tag = 0);

// Specializations for string
void sendString(const std::string& str, int dest_rank, int tag = 0);
std::string receiveString(int source_rank, int tag = 0);

// Specialization for pair
template<typename T1, typename T2>
void sendPair(const std::pair<T1, T2>& pair, int dest_rank, int tag = 0);

template<typename T1, typename T2>
std::pair<T1, T2> receivePair(int source_rank, int tag = 0);

// Generic send function that automatically handles any type
template <typename T>
void sendData(const T& data, int dest_rank, int tag = 0) {
    // Add this condition for C++11 compatibility
    #if __cplusplus >= 201703L
    if constexpr(std::is_trivially_copyable<T>::value) {
    #else
    if (std::is_trivially_copyable<T>::value) {
    #endif
        // Send trivially copyable individual item
        MPI_Send(&data, sizeof(T), MPI_BYTE, dest_rank, tag, MPI_COMM_WORLD);
    }
    #if __cplusplus >= 201703L
    else if constexpr(is_string<T>::value) {
    #else
    else if (is_string<T>::value) {
    #endif
        // Send string
        sendString(data, dest_rank, tag);
    }
    #if __cplusplus >= 201703L
    else if constexpr(is_pair<T>::value) {
    #else
    else if (is_pair<T>::value) {
    #endif
        // Send pair
        sendPair(data, dest_rank, tag);
    }
    #if __cplusplus >= 201703L
    else if constexpr(is_vector<T>::value) {
    #else
    else if (is_vector<T>::value) {
    #endif
        using value_type = typename T::value_type;
        
        #if __cplusplus >= 201703L
        if constexpr(std::is_trivially_copyable<value_type>::value) {
        #else
        if (std::is_trivially_copyable<value_type>::value) {
        #endif
            // Vector of trivially copyable types
            sendDataSimple(data, dest_rank, tag);
        }
        #if __cplusplus >= 201703L
        else if constexpr(is_vector<value_type>::value) {
        #else
        else if (is_vector<value_type>::value) {
        #endif
            // Nested vectors
            sendDataNested(data, dest_rank, tag);
        }
        else {
            // Vector of complex types
            int size = data.size();
            MPI_Send(&size, 1, MPI_INT, dest_rank, tag, MPI_COMM_WORLD);
            for (const auto& item : data) {
                sendData(item, dest_rank, tag + 1);
            }
        }
    }
    #if __cplusplus >= 201703L
    else if constexpr(is_map_like<T>::value) {
    #else
    else if (is_map_like<T>::value) {
    #endif
        // Map-like container (key-value pairs)
        int size = data.size();
        MPI_Send(&size, 1, MPI_INT, dest_rank, tag, MPI_COMM_WORLD);
        
        // Replace structured bindings for C++11 compatibility
        #if __cplusplus >= 201703L
        for (const auto& [key, value] : data) {
            sendData(key, dest_rank, tag + 1);
            sendData(value, dest_rank, tag + 2);
        }
        #else
        for (typename T::const_iterator it = data.begin(); it != data.end(); ++it) {
            sendData(it->first, dest_rank, tag + 1);
            sendData(it->second, dest_rank, tag + 2);
        }
        #endif
    }
    #if __cplusplus >= 201703L
    else if constexpr(is_container<T>::value) {
    #else
    else if (is_container<T>::value) {
    #endif
        // General container
        int size = data.size();
        MPI_Send(&size, 1, MPI_INT, dest_rank, tag, MPI_COMM_WORLD);
        for (const auto& item : data) {
            sendData(item, dest_rank, tag + 1);
        }
    }
    else {
        // Try to use the serialize method if available
        #if __cplusplus >= 201703L
        if constexpr(has_serialize_method<T>::value) {
        #else
        // For C++11, we'll handle this at compile time using SFINAE
        sendData_helper(data, dest_rank, tag, std::integral_constant<bool, has_serialize_method<T>::value>());
        #endif
            #if __cplusplus >= 201703L
            auto serialized = data.serialize();
            sendData(serialized, dest_rank, tag);
            #endif
        #if __cplusplus >= 201703L
        }
        else {
            // Complex non-container type (must be handled by user-defined serialization)
            static_assert(sizeof(T) == -1, "Unsupported type for automatic serialization");
        }
        #endif
    }
}

// Helper functions for C++11 compatibility
template <typename T>
void sendData_helper(const T& data, int dest_rank, int tag, std::true_type) {
    auto serialized = data.serialize();
    sendData(serialized, dest_rank, tag);
}

template <typename T>
void sendData_helper(const T&, int, int, std::false_type) {
    static_assert(sizeof(T) == -1, "Unsupported type for automatic serialization");
}

// Generic receive function that automatically handles any type
template <typename T>
T receiveData(int source_rank, int tag = 0) {
    MPI_Status status;
    
    #if __cplusplus >= 201703L
    if constexpr(std::is_trivially_copyable<T>::value) {
    #else
    if (std::is_trivially_copyable<T>::value) {
    #endif
        // Receive trivially copyable individual item
        T data;
        MPI_Recv(&data, sizeof(T), MPI_BYTE, source_rank, tag, MPI_COMM_WORLD, &status);
        return data;
    }
    #if __cplusplus >= 201703L
    else if constexpr(is_string<T>::value) {
    #else
    else if (is_string<T>::value) {
    #endif
        // Receive string
        return receiveString(source_rank, tag);
    }
    #if __cplusplus >= 201703L
    else if constexpr(is_pair<T>::value) {
    #else
    else if (is_pair<T>::value) {
    #endif
        // Receive pair
        return receivePair<typename T::first_type, typename T::second_type>(source_rank, tag);
    }
    #if __cplusplus >= 201703L
    else if constexpr(is_vector<T>::value) {
    #else
    else if (is_vector<T>::value) {
    #endif
        using value_type = typename T::value_type;
        
        #if __cplusplus >= 201703L
        if constexpr(std::is_trivially_copyable<value_type>::value) {
        #else
        if (std::is_trivially_copyable<value_type>::value) {
        #endif
            // Vector of trivially copyable types
            return receiveDataSimple<value_type>(source_rank, tag);
        }
        #if __cplusplus >= 201703L
        else if constexpr(is_vector<value_type>::value) {
        #else
        else if (is_vector<value_type>::value) {
        #endif
            // Nested vectors
            return receiveDataNested<typename value_type::value_type>(source_rank, tag);
        }
        else {
            // Vector of complex types
            int size;
            MPI_Recv(&size, 1, MPI_INT, source_rank, tag, MPI_COMM_WORLD, &status);
            T result;
            result.resize(size);
            for (int i = 0; i < size; i++) {
                result[i] = receiveData<value_type>(source_rank, tag + 1);
            }
            return result;
        }
    }
    #if __cplusplus >= 201703L
    else if constexpr(is_map_like<T>::value) {
    #else
    else if (is_map_like<T>::value) {
    #endif
        // Map-like container (key-value pairs)
        int size;
        MPI_Recv(&size, 1, MPI_INT, source_rank, tag, MPI_COMM_WORLD, &status);
        T result;
        for (int i = 0; i < size; i++) {
            typename T::key_type key = receiveData<typename T::key_type>(source_rank, tag + 1);
            typename T::mapped_type value = receiveData<typename T::mapped_type>(source_rank, tag + 2);
            result.emplace(std::move(key), std::move(value));
        }
        return result;
    }
    #if __cplusplus >= 201703L
    else if constexpr(is_container<T>::value) {
    #else
    else if (is_container<T>::value) {
    #endif
        // General container
        int size;
        MPI_Recv(&size, 1, MPI_INT, source_rank, tag, MPI_COMM_WORLD, &status);
        T result;
        for (int i = 0; i < size; i++) {
            auto item = receiveData<typename T::value_type>(source_rank, tag + 1);
            result.insert(result.end(), std::move(item));
        }
        return result;
    }
    else {
        // Try to use the deserialize method if available
        #if __cplusplus >= 201703L
        if constexpr(has_deserialize_method<T>::value) {
        #else
        // For C++11, we'll handle this at compile time with a helper function
        return receiveData_helper<T>(source_rank, tag, std::integral_constant<bool, has_deserialize_method<T>::value>());
        #endif
            #if __cplusplus >= 201703L
            auto serialized = receiveData<typename T::serialized_type>(source_rank, tag);
            return T::deserialize(serialized);
            #endif
        #if __cplusplus >= 201703L
        }
        else {
            // Complex non-container type (must be handled by user-defined deserialization)
            static_assert(sizeof(T) == -1, "Unsupported type for automatic deserialization");
            return T(); // Never reached, just to satisfy compiler
        }
        #endif
    }
    
    // Add this to avoid compiler warnings about missing return statement in C++11
    #if __cplusplus < 201703L
    return T();
    #endif
}

// Helper functions for C++11 compatibility
template <typename T>
T receiveData_helper(int source_rank, int tag, std::true_type) {
    auto serialized = receiveData<typename T::serialized_type>(source_rank, tag);
    return T::deserialize(serialized);
}

template <typename T>
T receiveData_helper(int, int, std::false_type) {
    static_assert(sizeof(T) == -1, "Unsupported type for automatic deserialization");
    return T(); // Never reached, just to satisfy compiler
}

// For trivially copyable types
template <typename T>
typename std::enable_if<std::is_trivially_copyable<T>::value, void>::type
sendDataSimple(const std::vector<T>& data, int dest_rank, int tag) {
    int count = data.size();
    MPI_Send(&count, 1, MPI_INT, dest_rank, tag, MPI_COMM_WORLD);
    if (count > 0) {
        MPI_Send(data.data(), count * sizeof(T), MPI_BYTE, dest_rank, tag + 1, MPI_COMM_WORLD);
    }
}

// Specialization for vectors of vectors (nested)
template <typename T>
void sendDataNested(const std::vector<std::vector<T>>& data, int dest_rank, int tag) {
    // Send number of vectors
    int outer_count = data.size();
    MPI_Send(&outer_count, 1, MPI_INT, dest_rank, tag, MPI_COMM_WORLD);
    
    // Send each inner vector
    for (const auto& vec : data) {
        int inner_count = vec.size();
        MPI_Send(&inner_count, 1, MPI_INT, dest_rank, tag + 1, MPI_COMM_WORLD);
        if (inner_count > 0) {
            MPI_Send(vec.data(), inner_count * sizeof(T), MPI_BYTE, dest_rank, tag + 2, MPI_COMM_WORLD);
        }
    }
}

// For trivially copyable types
template <typename T>
typename std::enable_if<std::is_trivially_copyable<T>::value, std::vector<T>>::type
receiveDataSimple(int source_rank, int tag) {
    int count;
    MPI_Status status;
    MPI_Recv(&count, 1, MPI_INT, source_rank, tag, MPI_COMM_WORLD, &status);
    std::vector<T> data(count);
    if (count > 0) {
        MPI_Recv(data.data(), count * sizeof(T), MPI_BYTE, source_rank, tag + 1, MPI_COMM_WORLD, &status);
    }
    return data;
}

// Specialization for vectors of vectors (nested)
template <typename T>
std::vector<std::vector<T>> receiveDataNested(int source_rank, int tag) {
    // Receive number of vectors
    int outer_count;
    MPI_Status status;
    MPI_Recv(&outer_count, 1, MPI_INT, source_rank, tag, MPI_COMM_WORLD, &status);
    
    std::vector<std::vector<T>> data(outer_count);
    
    // Receive each inner vector
    for (int i = 0; i < outer_count; i++) {
        int inner_count;
        MPI_Recv(&inner_count, 1, MPI_INT, source_rank, tag + 1, MPI_COMM_WORLD, &status);
        data[i].resize(inner_count);
        
        if (inner_count > 0) {
            MPI_Recv(data[i].data(), inner_count * sizeof(T), MPI_BYTE, 
                    source_rank, tag + 2, MPI_COMM_WORLD, &status);
        }
    }
    
    return data;
}

// String serialization
inline void sendString(const std::string& str, int dest_rank, int tag) {
    int length = str.length();
    MPI_Send(&length, 1, MPI_INT, dest_rank, tag, MPI_COMM_WORLD);
    if (length > 0) {
        MPI_Send(str.data(), length, MPI_CHAR, dest_rank, tag + 1, MPI_COMM_WORLD);
    }
}

inline std::string receiveString(int source_rank, int tag) {
    int length;
    MPI_Status status;
    MPI_Recv(&length, 1, MPI_INT, source_rank, tag, MPI_COMM_WORLD, &status);
    
    std::string str(length, '\0');
    if (length > 0) {
        MPI_Recv(&str[0], length, MPI_CHAR, source_rank, tag + 1, MPI_COMM_WORLD, &status);
    }
    return str;
}

// Pair serialization
template<typename T1, typename T2>
void sendPair(const std::pair<T1, T2>& pair, int dest_rank, int tag) {
    sendData(pair.first, dest_rank, tag);
    sendData(pair.second, dest_rank, tag + 100); // Using a different tag range to avoid collisions
}

template<typename T1, typename T2>
std::pair<T1, T2> receivePair(int source_rank, int tag) {
    T1 first = receiveData<T1>(source_rank, tag);
    T2 second = receiveData<T2>(source_rank, tag + 100); // Using a different tag range to avoid collisions
    return std::make_pair(first, second);
}

// Broadcast utility functions
template <typename T>
void broadcastData(T& data, int root) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (rank == root) {
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        for (int i = 0; i < size; i++) {
            if (i != root) {
                sendData(data, i);
            }
        }
    } else {
        data = receiveData<T>(root);
    }
}

// Scatter utility function for vectors
template <typename T>
std::vector<T> scatterVector(const std::vector<T>& data, int root) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    std::vector<T> local_data;
    
    if (rank == root) {
        // Calculate chunk sizes
        int total_size = data.size();
        int base_chunk_size = total_size / size;
        int remainder = total_size % size;
        
        int start_idx = 0;
        for (int i = 0; i < size; i++) {
            int chunk_size = base_chunk_size + (i < remainder ? 1 : 0);
            
            if (i == rank) {
                // Root's own chunk
                local_data.assign(data.begin() + start_idx, data.begin() + start_idx + chunk_size);
            } else {
                // Send chunk to other processes
                std::vector<T> chunk(data.begin() + start_idx, data.begin() + start_idx + chunk_size);
                sendData(chunk, i);
            }
            
            start_idx += chunk_size;
        }
    } else {
        // Receive chunk from root
        local_data = receiveData<std::vector<T>>(root);
    }
    
    return local_data;
}

// Gather utility function for vectors
template <typename T>
std::vector<T> gatherVector(const std::vector<T>& local_data, int root) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    std::vector<T> gathered_data;
    
    if (rank == root) {
        // Root collects data from all processes
        gathered_data = local_data;  // Start with root's own data
        
        for (int i = 0; i < size; i++) {
            if (i != root) {
                // Receive data from other processes
                auto received = receiveData<std::vector<T>>(i);
                gathered_data.insert(gathered_data.end(), received.begin(), received.end());
            }
        }
    } else {
        // Send local data to root
        sendData(local_data, root);
    }
    
    return gathered_data;
}

// --- Add after gatherVector and before #endif -------------------------------
#include "structs.h"

template<>
inline void sendData<Particle>(const Particle& data, int dest_rank, int tag) {
    MPI_Send(&data, sizeof(Particle), MPI_BYTE, dest_rank, tag, MPI_COMM_WORLD);
}

template<>
inline Particle receiveData<Particle>(int source_rank, int tag) {
    MPI_Status status;
    Particle data;
    MPI_Recv(&data, sizeof(Particle), MPI_BYTE, source_rank, tag, MPI_COMM_WORLD, &status);
    return data;
}
// ---------------------------------------------------------------------------

#endif // DATA_UTILS_H
