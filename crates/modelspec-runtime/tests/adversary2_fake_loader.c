/* Adversary pass 2 stand-in Vulkan loader for story:m1-inspect-device.
 * Three physical devices: 0 reports Vulkan 1.2.0, 1 reports Vulkan 1.1.0, 2 reports Vulkan 1.0.0.
 * Every device reports a subgroup size of 32 through each structure Vulkan 1.1 defines for it
 * (VkPhysicalDeviceSubgroupProperties, VkPhysicalDeviceVulkan11Properties), a device UUID through
 * VkPhysicalDeviceIDProperties, sane compute limits, one 1 GiB device-local heap and one compute
 * queue family. vkGetPhysicalDeviceProperties2 is answered for a 1.1 or 1.2 device only. */
#include <stdint.h>
#include <string.h>
#include <vulkan/vulkan_core.h>

static const uint32_t API[3] = { VK_MAKE_API_VERSION(0, 1, 2, 0), VK_MAKE_API_VERSION(0, 1, 1, 0),
                                 VK_MAKE_API_VERSION(0, 1, 0, 0) };
#define DEVICES 3u

static int index_of(VkPhysicalDevice device) { return (int)((uintptr_t)device - 0x2000); }

static VkResult create_instance(const VkInstanceCreateInfo *info, const VkAllocationCallbacks *alloc,
                                VkInstance *instance) {
    (void)info; (void)alloc; *instance = (VkInstance)(uintptr_t)0x1000; return VK_SUCCESS;
}
static void destroy_instance(VkInstance instance, const VkAllocationCallbacks *alloc) {
    (void)instance; (void)alloc;
}
static VkResult enumerate_physical_devices(VkInstance instance, uint32_t *count,
                                           VkPhysicalDevice *devices) {
    (void)instance;
    if (devices == 0) { *count = DEVICES; return VK_SUCCESS; }
    uint32_t n = *count < DEVICES ? *count : DEVICES;
    for (uint32_t i = 0; i < n; i++) devices[i] = (VkPhysicalDevice)(uintptr_t)(0x2000 + i);
    *count = n;
    return n < DEVICES ? VK_INCOMPLETE : VK_SUCCESS;
}
static void properties(VkPhysicalDevice device, VkPhysicalDeviceProperties *out) {
    int i = index_of(device);
    memset(out, 0, sizeof *out);
    out->apiVersion = API[i];
    out->driverVersion = 3;
    out->vendorID = 1;
    out->deviceID = 2 + (uint32_t)i;
    out->limits.maxComputeWorkGroupInvocations = 1024;
    out->limits.maxComputeWorkGroupSize[0] = 1024;
    out->limits.maxComputeWorkGroupSize[1] = 1024;
    out->limits.maxComputeWorkGroupSize[2] = 64;
    out->limits.maxComputeWorkGroupCount[0] = 65535;
    out->limits.maxComputeWorkGroupCount[1] = 65535;
    out->limits.maxComputeWorkGroupCount[2] = 65535;
    out->limits.maxComputeSharedMemorySize = 49152;
    out->limits.maxStorageBufferRange = 0x7fffffff;
    out->limits.minStorageBufferOffsetAlignment = 16;
    out->limits.timestampComputeAndGraphics = VK_TRUE;
}
static void properties2(VkPhysicalDevice device, VkPhysicalDeviceProperties2 *out) {
    properties(device, &out->properties);
    for (VkBaseOutStructure *s = (VkBaseOutStructure *)out->pNext; s != 0; s = s->pNext) {
        switch (s->sType) {
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES: {
            VkPhysicalDeviceVulkan11Properties *p = (VkPhysicalDeviceVulkan11Properties *)s;
            memset(p->deviceUUID, 0xab, VK_UUID_SIZE);
            p->subgroupSize = 32;
            p->subgroupSupportedStages = VK_SHADER_STAGE_COMPUTE_BIT;
            p->subgroupSupportedOperations = VK_SUBGROUP_FEATURE_BASIC_BIT;
            break;
        }
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES: {
            VkPhysicalDeviceSubgroupProperties *p = (VkPhysicalDeviceSubgroupProperties *)s;
            p->subgroupSize = 32;
            p->supportedStages = VK_SHADER_STAGE_COMPUTE_BIT;
            p->supportedOperations = VK_SUBGROUP_FEATURE_BASIC_BIT;
            break;
        }
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES: {
            VkPhysicalDeviceIDProperties *p = (VkPhysicalDeviceIDProperties *)s;
            memset(p->deviceUUID, 0xab, VK_UUID_SIZE);
            break;
        }
        default:
            break;
        }
    }
}
static void features2(VkPhysicalDevice device, VkPhysicalDeviceFeatures2 *out) {
    (void)device; (void)out;
}
static void memory(VkPhysicalDevice device, VkPhysicalDeviceMemoryProperties *out) {
    (void)device;
    memset(out, 0, sizeof *out);
    out->memoryHeapCount = 1;
    out->memoryHeaps[0].size = 1ull << 30;
    out->memoryHeaps[0].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
    out->memoryTypeCount = 1;
    out->memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
}
static void memory2(VkPhysicalDevice device, VkPhysicalDeviceMemoryProperties2 *out) {
    memory(device, &out->memoryProperties);
}
static VkResult device_extensions(VkPhysicalDevice device, const char *layer, uint32_t *count,
                                  VkExtensionProperties *out) {
    (void)device; (void)layer; (void)out; *count = 0; return VK_SUCCESS;
}
static void queue_families(VkPhysicalDevice device, uint32_t *count,
                           VkQueueFamilyProperties *out) {
    (void)device;
    if (out == 0) { *count = 1; return; }
    if (*count >= 1) {
        memset(out, 0, sizeof *out);
        out->queueFlags = VK_QUEUE_COMPUTE_BIT;
        out->queueCount = 1;
        out->timestampValidBits = 64;
        *count = 1;
    }
}
static VkResult create_device(VkPhysicalDevice physical, const VkDeviceCreateInfo *info,
                              const VkAllocationCallbacks *alloc, VkDevice *device) {
    (void)info; (void)alloc;
    *device = (VkDevice)(uintptr_t)(0x3000 + index_of(physical));
    return VK_SUCCESS;
}
static void destroy_device(VkDevice device, const VkAllocationCallbacks *alloc) {
    (void)device; (void)alloc;
}
static PFN_vkVoidFunction get_device_proc_addr(VkDevice device, const char *name) {
    (void)device;
    if (strcmp(name, "vkDestroyDevice") == 0) return (PFN_vkVoidFunction)destroy_device;
    return 0;
}
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance,
                                                               const char *name) {
    (void)instance;
    if (strcmp(name, "vkCreateInstance") == 0) return (PFN_vkVoidFunction)create_instance;
    if (strcmp(name, "vkDestroyInstance") == 0) return (PFN_vkVoidFunction)destroy_instance;
    if (strcmp(name, "vkEnumeratePhysicalDevices") == 0)
        return (PFN_vkVoidFunction)enumerate_physical_devices;
    if (strcmp(name, "vkGetPhysicalDeviceProperties") == 0) return (PFN_vkVoidFunction)properties;
    if (strcmp(name, "vkGetPhysicalDeviceProperties2") == 0) return (PFN_vkVoidFunction)properties2;
    if (strcmp(name, "vkGetPhysicalDeviceFeatures2") == 0) return (PFN_vkVoidFunction)features2;
    if (strcmp(name, "vkGetPhysicalDeviceMemoryProperties") == 0) return (PFN_vkVoidFunction)memory;
    if (strcmp(name, "vkGetPhysicalDeviceMemoryProperties2") == 0) return (PFN_vkVoidFunction)memory2;
    if (strcmp(name, "vkEnumerateDeviceExtensionProperties") == 0)
        return (PFN_vkVoidFunction)device_extensions;
    if (strcmp(name, "vkGetPhysicalDeviceQueueFamilyProperties") == 0)
        return (PFN_vkVoidFunction)queue_families;
    if (strcmp(name, "vkCreateDevice") == 0) return (PFN_vkVoidFunction)create_device;
    if (strcmp(name, "vkGetDeviceProcAddr") == 0) return (PFN_vkVoidFunction)get_device_proc_addr;
    return 0;
}
