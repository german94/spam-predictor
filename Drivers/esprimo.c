#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/init.h>
#include <linux/miscdevice.h>
#include <linux/module.h>
#include <linux/vmalloc.h>
#include <linux/time.h>
#include <asm/uaccess.h>
#include <linux/device.h>

// Variables globales
static int numero = 1;

bool esPrimo(int n)
{
    if (n<0) n = (-1)*n;

    if(n == 0 || n == 1)
        return false;

    int i;
    for(i = 2; i < n; i++)
    {
        if(n % i == 0)
            return false;
    }

    return true;
}

static ssize_t esprimo_write(struct file *file, const char __user *buf, size_t len, loff_t *ppos)
{
    char buffer[20];
    if (copy_from_user(buffer,buf,len))
        return -1;
    buffer[len] = 0;

    sscanf(buffer, "%d", &numero);

    return len;
}

//Funciones de lectura invocada por /dev fs
static ssize_t esprimo_read(struct file *file, char *buf, size_t count, loff_t *ppos)
{
    char buffer[2];
    int len = sprintf(buffer, "%d", esPrimo(numero));
    buffer[1] = 10;

    if (copy_to_user(buf, buffer,2))
        return -1;
    return len;
}

// Estructuras utililizadas por la funcion de registro de dispositivos
static const struct file_operations esprimo_fops = {
    .owner = THIS_MODULE,
    .read = esprimo_read,
    .write = esprimo_write
};

static struct miscdevice esprimo_dev = {
    MISC_DYNAMIC_MINOR,
    "esprimo",
    &esprimo_fops
};

// Funciones utilizadas por la creacion y destruccion del modulo
static int __init esprimo_init(void) {
    int ret;
    // Registración del device
    ret = misc_register(&esprimo_dev);
    if (ret)
        printk(KERN_ERR "No se puede registrar el dispositivo ESPRIMO\n");

    static struct class *esprimoVar;

    esprimoVar = class_create(THIS_MODULE, "esprimo");
    device_create(esprimoVar, NULL, 0, NULL, "esprimo");

    return ret;
}

static void __exit esprimo_exit(void) {
    misc_deregister(&esprimo_dev);
}

// Definicion de constructor y destructor del modulo.
module_init(esprimo_init);
module_exit(esprimo_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("ARG SA");
MODULE_DESCRIPTION("modulo que dice si un numero es primo o no");
